from fastapi import FastAPI, UploadFile, Form,File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Item CF Recommendation API")


def recommend_subjects(df: pd.DataFrame, student_id, top_n=5):
    # Проверяем нужные колонки
    if not all(col in df.columns for col in ["StudentID", "SubjectNameRU", "Grade"]):
        return {"error": "Файл должен содержать колонки StudentID, SubjectNameRU, Grade"}

    # Приводим StudentID к строке для единообразия
    df["StudentID"] = df["StudentID"].astype(str)
    student_id = str(student_id)

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
    # Чтение файла
    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file.file)
        else:
            df = pd.read_excel(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot read file: {e}")

    # Предобработка
    df = df[["StudentID", "SubjectNameRU", "Grade"]].dropna()
    df["Grade"] = pd.to_numeric(df["Grade"], errors="coerce")
    df = df.dropna(subset=["Grade"])
    df["StudentID"] = df["StudentID"].astype(str).str.strip()
    df["SubjectNameRU"] = df["SubjectNameRU"].astype(str).str.strip()

    if student_id not in df["StudentID"].values:
        raise HTTPException(status_code=404, detail="Student not found")

    # Пивотная таблица и косинусное сходство предметов
    pivot = df.pivot_table(index="StudentID", columns="SubjectNameRU", values="Grade", aggfunc="mean")
    pivot_filled = pivot.fillna(0)
    item_sim = pd.DataFrame(cosine_similarity(pivot_filled.T), index=pivot_filled.columns,
                            columns=pivot_filled.columns)

    # Курсы, которые студент уже прошёл
    taken = df[df["StudentID"] == student_id]["SubjectNameRU"].unique()
    not_taken = [c for c in pivot.columns if c not in taken]

    # Коллаборативная оценка
    student_grades = df[df["StudentID"] == student_id].groupby("SubjectNameRU")["Grade"].mean()
    cf_scores = {}
    for course in not_taken:
        sim_scores = item_sim[course].reindex(taken).dropna()
        if sim_scores.sum() == 0:
            cf_scores[course] = student_grades.mean()
        else:
            cf_scores[course] = sum(sim_scores[c] * student_grades.get(c, student_grades.mean()) for c in
                                    sim_scores.index) / sim_scores.sum()
    cf_df = pd.DataFrame(list(cf_scores.items()), columns=["SubjectNameRU", "cf_score"]).sort_values("cf_score",
                                                                                                     ascending=False).head(
        20)

    if cf_df.empty:
        return JSONResponse(content={"recommendations": []})

        # Признаки для модели XGBoost

    def feats(sid, course):
        sdata = df[df["StudentID"] == sid]
        sgrades = sdata.groupby("SubjectNameRU")["Grade"].mean()
        avg = sgrades.mean() if len(sgrades) > 0 else 3
        max_g = sgrades.max() if len(sgrades) > 0 else 3
        min_g = sgrades.min() if len(sgrades) > 0 else 3
        num = len(sgrades)
        sim_s = item_sim[course].reindex(
            sdata["SubjectNameRU"].unique()).dropna() if course in item_sim else pd.Series()
        cf = sum(sim_s[c] * sgrades.get(c, avg) for c in sim_s.index) / sim_s.sum() if len(sim_s) > 0 else avg
        cavg = df[df["SubjectNameRU"] == course]["Grade"].mean() if course in df["SubjectNameRU"].values else avg
        cpop = int(df[df["SubjectNameRU"] == course]["StudentID"].nunique())
        return [avg, max_g, min_g, cf, cavg, num, cpop]

        # Тренировка модели

    X_train, y_train = [], []
    for _, row in df.iterrows():
        sid, course, grade = row["StudentID"], row["SubjectNameRU"], row["Grade"]
        train_data = df[(df["StudentID"] != sid) | (df["SubjectNameRU"] != course)]
        X_train.append(feats(sid, course))
        y_train.append(grade)

    model = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                             random_state=42, verbosity=0)
    model.fit(np.array(X_train), np.array(y_train))

    # Предсказания
    predictions = []
    for course in cf_df["SubjectNameRU"]:
        pred = float(model.predict([feats(student_id, course)])[0])
        pred = round(np.clip(pred, 0, 4),2)
        predictions.append({"SubjectNameRU": course, "PredictedGrade": pred})

    # Топ-5
    top5 = sorted(predictions, key=lambda x: x["PredictedGrade"], reverse=True)[:5]
    return JSONResponse(content={"recommendations": top5})