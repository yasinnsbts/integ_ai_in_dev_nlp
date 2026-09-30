"""Задачи 4–6: эксперименты, выбор модели, сохранение и проверка загрузки.

Запуск из корня проекта: python train_model.py
Тот же код пошагово используется в notebooks/02_experiments_selection_saving.ipynb.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
SEED = 42
FOLDS = 5
SOURCE_COMMIT = "12d4c9e1cb42384d6b46e8110d819add4bfd9eac"
LABEL_NAMES = {
    "ACCOUNT_SEIZED": "Ограничения и блокировка счёта",
    "APP_LOGIN": "Проблемы со входом",
    "APP_TECH": "Технические проблемы приложения",
    "CARD_ISSUE": "Проблемы с картой",
    "FRAUD_SUSPECTED": "Подозрение на мошенничество",
    "INCOMING_DELAY": "Задержка входящего платежа",
    "KYC_VERIFICATION": "Проверка личности и документов",
    "PAYMENT_OUT_FAIL": "Проблемы с исходящим платежом",
}


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalized_text(texts: pd.Series) -> pd.Series:
    """Ключ для поиска повторов; не обучается на данных."""
    return texts.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


def load_data(root: Path = ROOT):
    """Читает готовое разбиение команды, исходные CSV не изменяет."""
    frames = []
    for name in ("train", "test"):
        path = root / "data" / "processed" / f"{name}.csv"
        df = pd.read_csv(path, encoding="utf-8-sig")
        if not {"text", "label"}.issubset(df.columns):
            raise ValueError(f"В {path} нужны столбцы text и label.")
        df = df[["text", "label"]].copy()
        if df.isna().any().any():
            raise ValueError(f"В {name} обнаружены пропуски.")
        if not df.map(lambda x: isinstance(x, str)).all().all():
            raise ValueError(f"В {name} текст и метка должны быть строками.")
        if df.apply(lambda col: col.str.strip().eq("").any()).any():
            raise ValueError(f"В {name} обнаружены пустые строки.")
        frames.append(df)
    train, test = frames
    if set(train.label) != set(test.label) or set(train.label) != set(LABEL_NAMES):
        raise ValueError("Ожидаются одни и те же 8 категорий проекта в train и test.")
    train_keys = normalized_text(train.text)
    test_keys = normalized_text(test.text)
    joined = pd.concat([train, test], ignore_index=True)
    conflicts = joined.groupby(normalized_text(joined.text)).label.nunique().gt(1)
    if conflicts.any():
        raise ValueError("У одинаковых нормализованных текстов разные метки; проверьте разметку.")
    # Дубли в train попадут в один fold. Пересечения с train исключаем
    # только из основной итоговой метрики; полный test тоже показываем.
    clean_test_mask = ~test_keys.isin(set(train_keys))
    if not clean_test_mask.any():
        raise ValueError("После исключения повторов train тестовая выборка пуста.")
    audit = {
        "train_rows": len(train),
        "test_rows": len(test),
        "classes": sorted(train.label.unique().tolist()),
        "normalization_for_duplicate_check": "lowercase, collapse whitespace, strip",
        "train_exact_duplicates": int(train.text.duplicated().sum()),
        "test_exact_duplicates": int(test.text.duplicated().sum()),
        "train_normalized_duplicates": int(train_keys.duplicated().sum()),
        "test_normalized_duplicates": int(test_keys.duplicated().sum()),
        "exact_overlap": len(set(train.text) & set(test.text)),
        "normalized_overlap_test_rows": int((~clean_test_mask).sum()),
        "primary_test_rows": int(clean_test_mask.sum()),
        "csv_sha256": {
            name: hashlib.sha256((root / "data" / "processed" / f"{name}.csv").read_bytes()).hexdigest()
            for name in ("train", "test")
        },
    }
    return train, test, train_keys, clean_test_mask, audit


def make_candidates() -> dict[str, Pipeline]:
    """Baseline плюс ровно три эксперимента. Параметры заданы до оценки test."""
    return {
        "baseline_word_lr": Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("classifier", LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)),
        ]),
        "exp1_word_bigrams_lr": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("classifier", LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)),
        ]),
        "exp2_word_bigrams_svc": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("classifier", LinearSVC(C=1.0, max_iter=5000, random_state=SEED)),
        ]),
        "exp3_char_svc": Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))),
            ("classifier", LinearSVC(C=1.0, max_iter=5000, random_state=SEED)),
        ]),
    }


def calculate_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def run_experiments(train, groups, candidates, output_dir: Path):
    """Все кандидаты используют одинаковые 5 folds и обучают TF-IDF внутри fold."""
    output_dir.mkdir(parents=True, exist_ok=True)
    splitter = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    splits = list(splitter.split(train.text, train.label, groups=groups))
    fold_ids = np.zeros(len(train), dtype=int)
    for fold, (fit_idx, val_idx) in enumerate(splits, start=1):
        if set(groups.iloc[fit_idx]) & set(groups.iloc[val_idx]):
            raise RuntimeError("Обнаружен повтор текста между обучением и валидацией.")
        if set(train.label.iloc[val_idx]) != set(train.label):
            raise ValueError("В одном из folds отсутствует класс; проверьте разбиение.")
        fold_ids[val_idx] = fold
    pd.DataFrame({"train_row": np.arange(len(train)), "fold": fold_ids}).to_csv(
        output_dir / "cv_folds.csv", index=False, encoding="utf-8-sig"
    )
    rows, out_of_fold = [], {}
    for name, candidate in candidates.items():
        predictions = np.empty(len(train), dtype=object)
        for fold, (fit_idx, val_idx) in enumerate(splits, start=1):
            model = clone(candidate)
            started = time.perf_counter()
            with threadpool_limits(limits=1):
                model.fit(train.text.iloc[fit_idx], train.label.iloc[fit_idx])
            fit_seconds = time.perf_counter() - started
            pred = model.predict(train.text.iloc[val_idx])
            predictions[val_idx] = pred
            rows.append({
                "model": name, "fold": fold, "fit_seconds": fit_seconds,
                **calculate_metrics(train.label.iloc[val_idx], pred),
            })
        out_of_fold[name] = predictions
        print(f"Завершено: {name}", flush=True)
    fold_metrics = pd.DataFrame(rows)
    summaries = []
    for name in candidates:
        subset = fold_metrics[fold_metrics.model == name]
        row = {"model": name}
        for metric in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "fit_seconds"]:
            row[f"{metric}_mean"] = float(subset[metric].mean())
            row[f"{metric}_std"] = float(subset[metric].std(ddof=0))
        summaries.append(row)
    # Правило выбора задаётся заранее: max mean F1-macro;
    # при точном равенстве — меньший std, затем имя для воспроизводимости.
    results = pd.DataFrame(summaries).sort_values(
        ["f1_macro_mean", "f1_macro_std", "model"], ascending=[False, True, True]
    ).reset_index(drop=True)
    results.to_csv(output_dir / "experiments.csv", index=False, encoding="utf-8-sig")
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    oof_table = train.copy()
    oof_table["fold"] = fold_ids
    for name, values in out_of_fold.items():
        oof_table[name] = values
    oof_table.to_csv(output_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    return results, out_of_fold


def choose_and_fit(results, candidates, train, output_dir: Path):
    best_name = str(results.iloc[0]["model"])
    write_json(output_dir / "selection.json", {
        "model": best_name,
        "rule": "max CV mean F1-macro, then min std, then model name",
        "test_used_for_selection": False,
        "cv_f1_macro_mean": float(results.iloc[0]["f1_macro_mean"]),
        "cv_f1_macro_std": float(results.iloc[0]["f1_macro_std"]),
    })
    model = clone(candidates[best_name])
    with threadpool_limits(limits=1):
        model.fit(train.text, train.label)
    print(f"Выбрана и обучена на всём train: {best_name}")
    return best_name, model


def evaluate_test(model, test, clean_test_mask, output_dir: Path):
    """Один прогноз выбранной модели: полный test и подмножество без повторов train."""
    pred = model.predict(test.text)
    clean = clean_test_mask.to_numpy()
    primary = calculate_metrics(test.label[clean], pred[clean])
    summary = {
        "primary_test_without_train_duplicates": {"n": int(clean.sum()), **primary},
        "original_test_all_rows": {"n": len(test), **calculate_metrics(test.label, pred)},
        "excluded_test_rows": np.flatnonzero(~clean).tolist(),
    }
    write_json(output_dir / "test_metrics.json", summary)
    report = classification_report(test.label[clean], pred[clean], output_dict=True, zero_division=0)
    write_json(output_dir / "classification_report.json", report)
    predictions = test.rename(columns={"label": "true_label"}).copy()
    predictions["predicted_label"] = pred
    predictions["included_in_primary_metric"] = clean
    predictions.to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    predictions.loc[clean & (test.label.to_numpy() != pred)].to_csv(
        output_dir / "test_errors.csv", index=False, encoding="utf-8-sig"
    )
    labels = sorted(model.classes_)
    matrix = confusion_matrix(test.label[clean], pred[clean], labels=labels)
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
        output_dir / "confusion_matrix.csv", index_label="true_label", encoding="utf-8-sig"
    )
    print("Итоговые метрики без повторов train:", json.dumps(primary, ensure_ascii=False))
    return pred, summary, report, matrix


def save_and_check(model, best_name, test, reference_pred, audit, summary, root: Path = ROOT):
    folder = root / "models"
    folder.mkdir(parents=True, exist_ok=True)
    # Pipeline содержит и обученный TF-IDF, и классификатор.
    joblib.dump(model, folder / "bank_classifier.joblib", compress=3)
    joblib.dump(model.named_steps["tfidf"], folder / "vectorizer.joblib", compress=3)
    joblib.dump(model.named_steps["classifier"], folder / "classifier.joblib", compress=3)
    restored = joblib.load(folder / "bank_classifier.joblib")
    vectorizer = joblib.load(folder / "vectorizer.joblib")
    classifier = joblib.load(folder / "classifier.joblib")
    pipeline_ok = np.array_equal(reference_pred, restored.predict(test.text))
    separate_ok = np.array_equal(reference_pred, classifier.predict(vectorizer.transform(test.text)))
    if not (pipeline_ok and separate_ok):
        raise RuntimeError("Прогнозы до сохранения и после загрузки различаются.")
    metadata = {
        "source_repository": "https://github.com/yasinnsbts/integ_ai_in_dev_nlp",
        "source_commit": SOURCE_COMMIT,
        "model_name": best_name,
        "pipeline": repr(model),
        "trained_on": "original data/processed/train.csv only",
        "train_rows": audit["train_rows"],
        "data_sha256": audit["csv_sha256"],
        "seed": SEED,
        "cv": "StratifiedGroupKFold, 5 folds; groups = lowercased whitespace-normalized text",
        "versions": {
            "python": platform.python_version(), "scikit-learn": sklearn.__version__,
            "numpy": np.__version__, "scipy": scipy.__version__,
            "pandas": pd.__version__, "joblib": joblib.__version__,
        },
        "labels": LABEL_NAMES,
        "test_metrics": summary,
        "round_trip": {"pipeline": bool(pipeline_ok), "separate_components": bool(separate_ok), "rows_checked": len(test)},
        "has_predict_proba": hasattr(model, "predict_proba"),
        "artifact_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(folder.glob("*.joblib"))
        },
    }
    write_json(folder / "metadata.json", metadata)
    print(f"Сохранение и обратная загрузка проверены на {len(test)} обращениях: OK")
    return metadata


def save_figures(results, matrix, labels, output_dir: Path):
    # Графики сохраняются в файлы; графическое окружение не требуется.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from sklearn.metrics import ConfusionMatrixDisplay

    fig = Figure(figsize=(10, 4.5))
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    ordered = results.iloc[::-1]
    ax.barh(ordered.model, ordered.f1_macro_mean, xerr=ordered.f1_macro_std, color="#3157a8", capsize=4)
    ax.set(xlim=(0, 1), xlabel="F1 macro (mean ± standard deviation)", title="5-fold cross-validation on train")
    fig.tight_layout()
    fig.savefig(output_dir / "cv_comparison.png", dpi=160)
    fig2 = Figure(figsize=(10, 8.5))
    FigureCanvasAgg(fig2)
    ax2 = fig2.subplots()
    ConfusionMatrixDisplay(matrix, display_labels=labels).plot(ax=ax2, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax2.set_title("Selected model: test without train duplicates")
    fig2.tight_layout()
    fig2.savefig(output_dir / "confusion_matrix.png", dpi=160)


def write_report(results, best_name, summary, report, audit, oof, train, metadata, output_dir: Path):
    baseline = results.set_index("model").loc["baseline_word_lr"]
    winner = results.set_index("model").loc[best_name]
    delta = 100 * (winner.f1_macro_mean - baseline.f1_macro_mean)
    primary = summary["primary_test_without_train_duplicates"]
    full = summary["original_test_all_rows"]
    base_oof = classification_report(train.label, oof["baseline_word_lr"], output_dict=True, zero_division=0)
    best_oof = classification_report(train.label, oof[best_name], output_dict=True, zero_division=0)
    lines = [
        "# Эксперименты, выбор и сохранение модели",
        "",
        f"Исходный репозиторий: https://github.com/yasinnsbts/integ_ai_in_dev_nlp, commit `{SOURCE_COMMIT}`.",
        "Задача: по тексту обращения выбрать одну из 8 категорий банковской поддержки.",
        "",
        "## Протокол и данные",
        "",
        f"Использованы готовые CSV команды: train — {audit['train_rows']}, test — {audit['test_rows']}. Исходные файлы не изменены.",
        f"После приведения к нижнему регистру и удаления лишних пробелов обнаружены {audit['train_normalized_duplicates']} повтора внутри train и {audit['normalized_overlap_test_rows']} строка test, совпадающая с train.",
        "TF-IDF сам приводит текст к нижнему регистру, поэтому такие совпадения существенны.",
        "Для сравнения используется StratifiedGroupKFold: 5 общих для всех моделей folds, seed=42; одинаковые нормализованные тексты всегда в одном fold.",
        "TF-IDF обучается только на обучающей части каждого fold, внутри Pipeline. Метки test не используются для выбора модели или параметров.",
        "Основной критерий — средний F1-macro; при точном равенстве выбирается меньший std, затем имя модели. Std рассчитан с ddof=0 и не является доверительным интервалом.",
        "В ноутбуке команды baseline получал F1-macro ≈0.912 при обычном StratifiedKFold. Ниже baseline пересчитан на групповых folds, поэтому его нельзя напрямую сравнивать со старым числом.",
        "",
        "## Задача 4. Три эксперимента и контрольный baseline",
        "",
        "| Вариант | Признаки TF-IDF | Классификатор | Проверяемое изменение |",
        "|---|---|---|---|",
        "| baseline_word_lr | Отдельные слова | LogisticRegression, C=1 | Контрольный уровень из проекта |",
        "| exp1_word_bigrams_lr | Слова и пары соседних слов | LogisticRegression, C=1 | Добавление контекста относительно baseline |",
        "| exp2_word_bigrams_svc | Слова и пары соседних слов | LinearSVC, C=1 | Смена классификатора относительно эксперимента 1 |",
        "| exp3_char_svc | char_wb, фрагменты 3–5 символов | LinearSVC, C=1 | Смена признаков относительно эксперимента 2 |",
        "",
        "Символьные признаки могут учитывать общие части русских слов и опечатки. Это гипотеза о признаках; отдельного теста устойчивости к опечаткам здесь нет.",
        "У всех TF-IDF остальные параметры стандартные; у LR max_iter=1000, у SVC max_iter=5000; seed=42.",
        "",
        "| Вариант | Accuracy | Precision macro | Recall macro | F1 macro ± std | Среднее время fit, с |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in results.itertuples():
        lines.append(f"| {row.model} | {row.accuracy_mean:.4f} | {row.precision_macro_mean:.4f} | {row.recall_macro_mean:.4f} | {row.f1_macro_mean:.4f} ± {row.f1_macro_std:.4f} | {row.fit_seconds_mean:.3f} |")
    lines += [
        "", "![Сравнение моделей](cv_comparison.png)", "",
        "## Задача 5. Выбор модели", "",
        f"Выбран `{best_name}`: наибольший средний F1-macro на общей кросс-валидации — {winner.f1_macro_mean:.4f}. Изменение относительно пересчитанного baseline: {delta:+.2f} процентного пункта.",
        "Macro-усреднение даёт каждой категории одинаковый вес. Это полезно, в частности, для менее многочисленного APP_TECH.",
        f"На OOF-прогнозах train F1 класса APP_TECH: baseline {base_oof['APP_TECH']['f1-score']:.4f}, выбранная модель {best_oof['APP_TECH']['f1-score']:.4f}; recall: {base_oof['APP_TECH']['recall']:.4f} → {best_oof['APP_TECH']['recall']:.4f}.",
        "OOF-метрики по объединённым прогнозам могут немного отличаться от среднего метрик по folds. Разница CV сама по себе не доказывает статистическую значимость.",
        "Выбранная модель затем обучена на всех строках train. Test не добавлялся к обучению.",
        "",
        "### Финальная проверка выбранной модели", "",
        "| Выборка | N | Accuracy | Precision macro | Recall macro | F1 macro |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Основная: test без совпадений с train | {primary['n']} | {primary['accuracy']:.4f} | {primary['precision_macro']:.4f} | {primary['recall_macro']:.4f} | {primary['f1_macro']:.4f} |",
        f"| Исходный test целиком, для сопоставимости | {full['n']} | {full['accuracy']:.4f} | {full['precision_macro']:.4f} | {full['recall_macro']:.4f} | {full['f1_macro']:.4f} |",
        "",
        "Обе строки рассчитаны из одного набора прогнозов одной выбранной модели; это не дополнительный подбор по test.",
        "",
        "| Класс | Precision | Recall | F1 | Количество |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in sorted(LABEL_NAMES):
        row = report[label]
        lines.append(f"| {label} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1-score']:.4f} | {int(row['support'])} |")
    lines += [
        "", "![Матрица ошибок](confusion_matrix.png)", "",
        "Ошибочные обращения перечислены в test_errors.csv; все прогнозы — в test_predictions.csv.",
        "",
        "## Задача 6. Сохранение и обратная загрузка", "",
        "Сохранены models/bank_classifier.joblib (весь Pipeline), models/vectorizer.joblib и models/classifier.joblib (отдельные компоненты), а также models/metadata.json с версиями, параметрами, метриками и SHA-256.",
        f"После joblib.load прогнозы всего Pipeline и отдельно vectorizer + classifier совпали с исходными на всех {metadata['round_trip']['rows_checked']} строках test.",
        "Для новых обращений достаточно model.predict([text]); повторный fit/fit_transform не нужен.",
        "LinearSVC не выдаёт вероятности через predict_proba. Его decision_function нельзя называть вероятностью без отдельной калибровки.",
        "",
        "## Ограничения", "",
        "Это оценка на небольшом учебном датасете. Происхождение и лицензия исходных обращений в просмотренных файлах проекта не описаны. Есть похожие формулировки; группировка по регистру и пробелам не устраняет все перефразировки и возможные шаблонные повторы.",
        "Высокие метрики не гарантируют такое же качество на реальных обращениях. Категории вне этих восьми, несколько тем в одном обращении и автоматическое направление в банковские очереди здесь не реализованы.",
        "Результаты test не использовались для изменения кандидатов. Повторный запуск на тех же файлах проверяет воспроизводимость, но не является новой независимой оценкой.",
        "",
        "## Воспроизведение", "",
        "Установить requirements-experiments.txt и выполнить python train_model.py либо Run All в новом ноутбуке. Время fit зависит от компьютера.",
        f"Среда этого прогона: Python {platform.python_version()}, scikit-learn {sklearn.__version__}, pandas {pd.__version__}, numpy {np.__version__}, scipy {scipy.__version__}, joblib {joblib.__version__}.",
        "",
    ]
    (output_dir / "model_selection.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    output_dir = ROOT / "reports" / "experiments"
    train, test, groups, clean_mask, audit = load_data()
    write_json(output_dir / "data_audit.json", audit)
    candidates = make_candidates()
    results, oof = run_experiments(train, groups, candidates, output_dir)
    print(results[["model", "accuracy_mean", "f1_macro_mean", "f1_macro_std"]].to_string(index=False))
    best_name, model = choose_and_fit(results, candidates, train, output_dir)
    pred, summary, report, matrix = evaluate_test(model, test, clean_mask, output_dir)
    metadata = save_and_check(model, best_name, test, pred, audit, summary)
    save_figures(results, matrix, sorted(model.classes_), output_dir)
    write_report(results, best_name, summary, report, audit, oof, train, metadata, output_dir)
    print("Готово: models/ и reports/experiments/")


if __name__ == "__main__":
    main()
