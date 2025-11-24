# ---------------------------------------------------
# 🧠 GEN-AI STYLE QUESTION ANSWERING MODULE
# ---------------------------------------------------

st.sidebar.markdown("### 🧠 Ask a Pollution Question")
user_question = st.sidebar.text_input("Ask any question (e.g., 'Which place has highest NO2?')")

def answer_pollution_question(question, df):
    """Simple Gen-AI style answering without external API."""
    question_lower = question.lower()
    pol = pollutant

    # Highest pollution / hotspot
    if "highest" in question_lower or "hotspot" in question_lower or "high" in question_lower:
        row = df.loc[df[pol].idxmax()]
        return (
            f"🔥 **Highest {pol} level** detected near:\n\n"
            f"• **Latitude:** {row['lat']:.3f}\n"
            f"• **Longitude:** {row['lon']:.3f}\n"
            f"• **Value:** {row[pol]:.2f}\n\n"
            f"This region is the **primary pollution hotspot** in your selected time period."
        )

    # Lowest pollution / cleanest area
    if "lowest" in question_lower or "cleanest" in question_lower or "low" in question_lower:
        row = df.loc[df[pol].idxmin()]
        return (
            f"🌿 **Lowest {pol} level** detected near:\n\n"
            f"• **Latitude:** {row['lat']:.3f}\n"
            f"• **Longitude:** {row['lon']:.3f}\n"
            f"• **Value:** {row[pol]:.2f}\n\n"
            f"This area appears to be **relatively cleaner**."
        )

    # Average pollution
    if "average" in question_lower or "mean" in question_lower:
        val = df[pol].mean()
        return f"📊 The **average {pol} level** in the selected region/time is **{val:.2f}**."

    # Trend
    if "trend" in question_lower or "increasing" in question_lower or "decreasing" in question_lower:
        df_temp = df.copy()
        df_temp["year"] = df_temp["date"].dt.year
        trend = df_temp.groupby("year")[pol].mean()

        if len(trend) < 2:
            return "⚠️ Not enough yearly data to detect trends."

        if trend.iloc[-1] > trend.iloc[0]:
            return "📈 Pollution shows an **increasing trend** over the years."
        else:
            return "📉 Pollution shows a **decreasing trend** over the years."

    # Unsupported question
    return "❓ I couldn't understand the question. Try asking about **highest**, **lowest**, **average**, or **trend**."
