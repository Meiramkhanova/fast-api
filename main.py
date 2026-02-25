from fastapi import FastAPI, UploadFile, Form,File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Item CF Recommendation API")


def recommend_subjects(df: pd.DataFrame, student_id, top_n=5):
    # Проверяем нужные колонки
    if not all(col in df.columns for col in ["StudentID", "SubjectNameRU", "Grade"]):
        return {"error": "Файл должен содержать колонки StudentID, SubjectNameRU, Grade"}

    # Приводим StudentID к строке и чистим формат
    df["StudentID"] = (
        df["StudentID"]
        .astype(str)
        .str.strip()
        .str.replace(".0", "", regex=False)
        .str.replace(" ", "")
    )

    student_id = str(student_id).strip().replace(" ", "")

    # Пивот таблица
    pivot = df.pivot_table(
        index="StudentID",
        columns="SubjectNameRU",
        values="Grade",
        aggfunc="mean"
    ).fillna(0)

    if student_id not in pivot.index:
        return {"error": "Студент не найден"}

    user_vector = pivot.loc[student_id]
    taken_subjects = user_vector[user_vector > 0].index

    if len(taken_subjects) == 0:
        return {"error": "Нет оценок у студента"}

    # Косинусная похожесть предметов
    item_similarity = pd.DataFrame(
        cosine_similarity(pivot.T),
        index=pivot.columns,
        columns=pivot.columns
    )

    # Средняя похожесть для рекомендаций
    scores = item_similarity[taken_subjects].mean(axis=1)
    scores = scores.drop(taken_subjects, errors="ignore")
    recommendations = scores.sort_values(ascending=False).head(top_n)

    # Преобразуем в DataFrame и в JSON
    df_result = recommendations.reset_index()
    df_result.columns = ["SubjectNameRU", "PredictedGrade"]

    if df_result.empty:
        return {"error": "Нет рекомендаций для этого студента"}

    return df_result.to_dict(orient="records")


@app.post("/recommend")
async def recommend(file: UploadFile, student_id: str = Form(...)):
    try:
        if file.filename.endswith(".xlsx"):
            df = pd.read_excel(file.file)
        elif file.filename.endswith(".csv"):
            df = pd.read_csv(file.file)
        else:
            return JSONResponse(
                content={"error": "Поддерживаются только .xlsx или .csv"},
                status_code=400
            )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

    result = recommend_subjects(df, student_id)
    return JSONResponse(content=result)

@app.post("/recommendsecond/")
async def recommend(student_id: str, file: UploadFile = File(...)):

    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file.file)
        else:
            df = pd.read_excel(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot read file: {e}")

    # ===============================
    # PREPROCESSING
    # ===============================

    df = df[["StudentID", "SubjectNameRU", "Grade"]].dropna()

    df["Grade"] = pd.to_numeric(df["Grade"], errors="coerce")
    df = df.dropna(subset=["Grade"])

    df["StudentID"] = df["StudentID"].astype(str).str.strip()
    df["SubjectNameRU"] = df["SubjectNameRU"].astype(str).str.strip()

    if student_id not in df["StudentID"].values:
        raise HTTPException(status_code=404, detail="Student not found")

    # ===============================
    # FAST FEATURE ENGINEERING
    # ===============================

    df["student_mean"] = df.groupby("StudentID")["Grade"].transform("mean")
    df["student_max"] = df.groupby("StudentID")["Grade"].transform("max")
    df["student_min"] = df.groupby("StudentID")["Grade"].transform("min")
    df["student_count"] = df.groupby("StudentID")["Grade"].transform("count")

    df["course_mean"] = df.groupby("SubjectNameRU")["Grade"].transform("mean")
    df["course_pop"] = df.groupby("SubjectNameRU")["StudentID"].transform("nunique")

    # ===============================
    # PIVOT + SIMILARITY
    # ===============================

    pivot = df.pivot_table(
        index="StudentID",
        columns="SubjectNameRU",
        values="Grade",
        aggfunc="mean"
    ).fillna(0)

    similarity_matrix = pd.DataFrame(
        cosine_similarity(pivot.T),
        index=pivot.columns,
        columns=pivot.columns
    )

    taken_courses = df[df["StudentID"] == student_id]["SubjectNameRU"].unique()
    all_courses = pivot.columns.tolist()
    not_taken = [c for c in all_courses if c not in taken_courses]

    # ===============================
    # TRAIN MODEL
    # ===============================

    feature_cols = [
        "student_mean",
        "student_max",
        "student_min",
        "student_count",
        "course_mean",
        "course_pop"
    ]

    train_df = df.copy()

    X = train_df[feature_cols]
    y = train_df["Grade"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )

    model.fit(X_train, y_train)

    # ===============================
    # RECOMMENDATIONS
    # ===============================

    student_row = df[df["StudentID"] == student_id].iloc[0]

    recs = []

    for course in not_taken[:20]:

        temp = pd.DataFrame([{
            "student_mean": student_row["student_mean"],
            "student_max": student_row["student_max"],
            "student_min": student_row["student_min"],
            "student_count": student_row["student_count"],
            "course_mean": df[df["SubjectNameRU"] == course]["Grade"].mean(),
            "course_pop": df[df["SubjectNameRU"] == course]["StudentID"].nunique()
        }])

        pred = float(model.predict(temp)[0])
        pred = round(np.clip(pred, 0, 4), 2)

        recs.append({
            "SubjectNameRU": course,
            "PredictedGrade": pred
        })

    top5 = (
        pd.DataFrame(recs)
        .sort_values("PredictedGrade", ascending=False)
        .head(5)
        .to_dict(orient="records")
    )

    return JSONResponse(content={"recommendations": top5})