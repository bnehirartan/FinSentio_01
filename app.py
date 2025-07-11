
from datetime import datetime
import gradio as gr
import database as db
from financial_assistant import generate_financial_response 
from riskanalyzer import get_response
from llm_cost_calculator import LLMCostCalculator
import json
import pandas as pd
import uuid
from riskanalyzer import format_sentiment_response
import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# Global user session state
current_user = None
cost_calculator = LLMCostCalculator()

def generate_risk_pdf(risk_factors_table, summary, risk_level, risk_score, stability_score, user_profile, plot_path, vol, trend, lang="en"):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs("outputs", exist_ok=True)
    pdf_path = f"outputs/risk_report_{timestamp}.pdf"
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    def draw_wrapped_text(canvas_obj, text, x, y, max_width, font="Helvetica", size=10, leading=14):
        from reportlab.pdfbase.pdfmetrics import stringWidth
        canvas_obj.setFont(font, size)
        lines = []
        current_line = ""
        for word in text.split():
            if stringWidth(current_line + word + " ", font, size) < max_width:
                current_line += word + " "
            else:
                lines.append(current_line.strip())
                current_line = word + " "
        lines.append(current_line.strip())
        for line in lines:
            canvas_obj.drawString(x, y, line)
            y -= leading
        return y

    # Başlık
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 50, "Investment Risk Report" if lang == "en" else "Yatırım Risk Raporu")
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 65, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    y = height - 90

    # Kullanıcı Profili
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "User Profile")
    y -= 15
    c.setFont("Helvetica", 10)
    for k, v in user_profile.items():
        label = k.replace("_", " ").capitalize()
        value = v.replace("_", " ").capitalize() if isinstance(v, str) else str(v)
        c.drawString(60, y, f"{label}: {value}")
        y -= 13

    profile_line_count = len(user_profile)
    y -= profile_line_count * 13  # her satırın boyu
    y -= 20  # ekstra boşluk


    # Risk Faktörleri Tablosu
    table_data = [["Factor", "Status", "Comment"]] + [
        [
            row.get("Factor", "").replace("_", " "),
            row.get("Status", "").replace("_", " "),
            row.get("Comment", "").replace("_", " ")
        ]
        for row in risk_factors_table
    ]
    

   # Stil ayarları
    styles = getSampleStyleSheet()

    # Tablo verilerini wrap’li şekilde hazırla
    table_data = [["Factor", "Status", "Comment"]] + [
        [
            Paragraph(row["Factor"], styles["BodyText"]),
            Paragraph(row["Status"], styles["BodyText"]),
            Paragraph(row["Comment"], styles["BodyText"]),
        ]
        for row in risk_factors_table
    ]

    # Kolon genişlikleri (A4 sayfasına sığacak şekilde)
    col_widths = [100, 100, 330]

    # Tabloyu oluştur
    table = Table(table_data, colWidths=col_widths)

# Tablo stili
    table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.black),
    ("BOX", (0, 0), (-1, -1), 0.25, colors.black),
]))

    table.wrapOn(c, width, height)
    table.drawOn(c, 50, y - (18 * len(table_data)))
    y = y - (18 * len(table_data)) - 30

    # Grafik
    if plot_path and os.path.exists(plot_path):
        try:
            img = ImageReader(plot_path)
            c.drawImage(img, 50, y - 130, width=250, height=120)
        except Exception as img_err:
            c.drawString(50, y, f"[X] Failed to load chart image: {str(img_err)}")
        y -= 140

    # Volatility & Trend Insight
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Volatility & Trend Insight")
    y -= 15
    insight = f"Volatility is {vol}%, indicating {'low' if vol < 10 else 'high'} short-term risk. Trend is {trend}%, indicating {'positive' if trend >= 0 else 'negative'} momentum."
    y = draw_wrapped_text(c, insight, 50, y, width - 100)

   # Summary
    y -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Summary")
    y -= 15

    # Summary string olarak gelsin (JSON değil)
    # JSON gelirse temizle
    if isinstance(summary, str):
        if summary.strip().startswith("{"):
            try:
                parsed = json.loads(summary)
                summary_text = parsed.get("summary", "")
            except:
                summary_text = summary
        else:
            summary_text = summary
    else:
        summary_text = str(summary)

    # Gereksiz ```json``` veya kod bloklarını sil
    summary_text = summary_text.replace("```json", "").replace("```", "").strip()


    # Yazıyı düzgün şekilde sararak yazdır
    y = draw_wrapped_text(c, summary_text, 50, y, width - 100)



    c.save()
    return pdf_path


def login(username, password):
    """Login function for the interface"""
    global current_user
    success, result = db.authenticate_user(username, password)
    
    if success:
        current_user = result
        # update the current_user_id in the risk analyzer module
        import riskanalyzer
        riskanalyzer.current_user_id = current_user
        
        # try to load user profile
        profile_success, profile_data = db.get_user_profile(current_user)
        if profile_success:
            # user has an existing profile, we'll need to populate form in the load_profile function
            return "Login successful!", gr.update(visible=False), gr.update(visible=True)
        else:
            # user doesn't have a profile yet
            return "Login successful! Please complete your profile.", gr.update(visible=False), gr.update(visible=True)
    else:
        return f"Login failed: {result}", gr.update(visible=True), gr.update(visible=False)

def register(username, password, email, confirm_password):
    """Register function for the interface"""
    if password != confirm_password:
        return "Passwords do not match", gr.update(visible=True), gr.update(visible=False)
    
    success, message = db.register_user(username, password, email)
    
    if success:
        return f"{message}. Please login.", gr.update(visible=True), gr.update(visible=False)
    else:
        return message, gr.update(visible=True), gr.update(visible=False)

def logout():
    """Logout function for the interface"""
    global current_user
    current_user = None
    # Risk analyzer modülündeki current_user_id'yi sıfırla
    import riskanalyzer
    riskanalyzer.current_user_id = None
    return gr.update(visible=True), gr.update(visible=False)

def save_profile(risk_taker, risk_word, game_show, investment_allocation, 
                market_follow, new_investment, buy_things, finance_reading,
                previous_investments, investment_goal):
    """Save user profile information to the database"""
    if current_user is None:
        return "Error: User not logged in"
    
    # Checkbox groups için özel işlem, liste şeklinde geliyorlar
    previous_investments_list = previous_investments if isinstance(previous_investments, list) else []
    
    success, message = db.save_user_profile(
        current_user,
        risk_taker, risk_word, game_show, investment_allocation,
        market_follow, new_investment, buy_things, finance_reading,
        previous_investments_list, investment_goal
    )
    
    if success:
        return "Profile saved successfully!"
    else:
        return f"Error saving profile: {message}"

def load_profile():
    """Load user profile from database"""
    if current_user is None:
        return (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), 
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update())
    
    success, profile_data = db.get_user_profile(current_user)
    
    if not success:
        return (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), 
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update())
    
    # Return updates for all fields
    return (
        gr.update(value=profile_data.get('risk_taker', 'Cautious')),
        gr.update(value=profile_data.get('risk_word', 'Uncertainty')),
        gr.update(value=profile_data.get('game_show', '$1,000 in cash')),
        gr.update(value=profile_data.get('investment_allocation', '60% in low-risk, 30% in medium-risk, 10% in high-risk investments')),
        gr.update(value=profile_data.get('market_follow', 'Weekly')),
        gr.update(value=profile_data.get('new_investment', 'Research thoroughly before investing')),
        gr.update(value=profile_data.get('buy_things', 'Neutral')),
        gr.update(value=profile_data.get('finance_reading', 'Neutral')),
        gr.update(value=profile_data.get('previous_investments', [])),
        gr.update(value=profile_data.get('investment_goal', 'Long-term savings'))
    )

def education_chatbot(message, pdf_file, history):
    """Financial education chatbot using the financial_assistant module"""
    try:
        # Chat history'yi generate_financial_response fonksiyonuna geçir
        response = generate_financial_response(message, pdf_file, history)
        
        # LLM maliyetini hesapla - Gemini 2.0 Flash kullan
        cost_info = cost_calculator.calculate_cost(message, response, "gemini-2.0-flash")
        
        # HTML içeriğini Gradio chatbot için düzenle
        processed_response = response
        
        # Eğer içerik HTML içeriyorsa, bazı tagları temizle
        if "<div" in response or "<a" in response:
            # Link yapısını daha basit hale getir ama fonksiyonelliği koru
            processed_response = response.replace('<div style=\'font-family: sans-serif; line-height: 1.6\'>', '')
            processed_response = processed_response.replace('</div>', '')
            processed_response = processed_response.replace('<br><br><b>Kaynaklar:</b><br>', '\n\nKaynaklar:\n')
            processed_response = processed_response.replace('<br>', '\n')
            
            # Linkleri düzenle
            import re
            links = re.findall(r'<a href="([^"]+)" target="_blank">([^<]+)</a>', processed_response)
            for link_url, link_text in links:
                processed_response = processed_response.replace(f'<a href="{link_url}" target="_blank">{link_text}</a>', link_url)
        
        # Agent yönlendirme bilgilerini al
        from financial_assistant import danger_agent, intent_classifier, relevance_agent
        
        # Danger check
        danger_result = danger_agent.generate_response(message)
        danger_status = "🟢" if danger_result == "safe" else "🔴"
        
        # Intent check
        intent_result = intent_classifier.generate_response(message)
        
        # Relevance check
        relevance_result = relevance_agent.generate_response(message)
        
        # Agent yönlendirme bilgilerini oluştur
        agent_routing = f"""
### Agent Yönlendirme Bilgisi
{danger_status} Prompt is {danger_result.upper()} (DangerAgent)
🔀 Detected intent: {intent_result} (IntentClassifier)
🌐 Routed to {intent_result.upper()}Agent
"""
        
        # Maliyet bilgisini oluştur
        cost_message = f"""
{agent_routing}
### Maliyet Bilgisi (Gemini 2.0 Flash)
- *Prompt Token Sayısı:* {cost_info['prompt_tokens']}
- *Response Token Sayısı:* {cost_info['response_tokens']}
- *Toplam Token:* {cost_info['prompt_tokens'] + cost_info['response_tokens']}
- *Prompt Maliyeti:* ${cost_info['prompt_cost']:.6f}
- *Response Maliyeti:* ${cost_info['response_cost']:.6f}
- *Toplam Maliyet:* ${cost_info['total_cost']:.6f}
"""
        
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": processed_response})
        return history, cost_message

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Education chatbot hata: {e}\n{error_details}")
        return history + [[message, f"Bir hata oluştu: {str(e)}"]], "Maliyet hesaplanamadı"

def advisor_chatbot(message, history):
    """Risk & Analyzer chatbot using the riskanalyzer module"""
    try:
        response, intent, danger_result = get_response(message)
        # Sentiment detection

        pdf_file = None

        if isinstance(response, dict) and "risk_factors_table" in response:

            formatted_response = "📊 Risk Analysis Results:\n\n"
            formatted_response += "Risk Factors Table:\n"
            for row in response["risk_factors_table"]:
                formatted_response += f"• {row.get('Factor')}: {row.get('Status')} – {row.get('Comment')}\n"

            # 📈 Volatility & Trend grafik çizimi
            import matplotlib.pyplot as plt
            from datetime import datetime
            import uuid
            import os

            vol = float(response.get("volatility_percent", 0))
            trend = float(response.get("trend_percent", 0))

            os.makedirs("outputs", exist_ok=True)
            plot_path = os.path.join("outputs", f"plot_{uuid.uuid4().hex}.png")

            plt.figure()
            plt.bar(["Volatility", "Trend"], [vol, trend], color=["red", "orange"])
            plt.title("Volatility & Price Trend (%)")
            plt.ylabel("%")
            plt.savefig(plot_path)
            plt.close()

            # 👤 Kullanıcı profili ve dil
            from riskanalyzer import get_current_profile, danger_agent, analyze_prompt
            user_profile = get_current_profile()
            lang = "tr" if "ş" in message.lower() or "ı" in message.lower() else "en"

            # ✍️ LLM ile sade summary üret
            from financial_assistant import gemini
            sector_comment = response.get("sector_comment", "")
            macro_comment = response.get("macro_comment", "")
            user_profile_str = ", ".join([f"{k.replace('_', ' ').title()}: {v.replace('_', ' ').title()}" for k, v in user_profile.items()])

            import langdetect
            try:
                lang = langdetect.detect(message)
            except:
                lang = "en"  # Fallback


            summary_prompt = f"""
You are a financial assistant. Write a short, single-paragraph summary of the investment risk based on the data below. 
The user is a beginner investor, so your explanation should be simple, clear, and free of financial jargon. 
Do not use JSON, raw data structures, or tables.

Explain what each factor means in layman's terms, and connect the insights to how the user might make investment decisions. 
Emphasize overall stability or risk.

### Inputs:
- Volatility: {vol}%
- Trend: {trend}%
- Sector Analysis: {sector_comment}
- Macro Environment: {macro_comment}
- User Profile: {user_profile_str}
"""
            llm_summary = gemini.generate_content(summary_prompt).text.strip()

            # PDF üretimi (llm_summary JSON değil!)
            pdf_path = generate_risk_pdf(
                risk_factors_table=response["risk_factors_table"],
                summary=llm_summary,
                risk_level=response.get("risk_level", "Unknown"),
                risk_score=response.get("risk_score", 0),
                stability_score=response.get("stability_score", 0),
                user_profile=user_profile,
                plot_path=plot_path,
                vol=vol,
                trend=trend,
                lang=lang
            )
            pdf_file = pdf_path

            # LLM maliyet bilgisi
            cost_info = cost_calculator.calculate_cost(message, llm_summary, "gemini-1.5-flash-latest")

            # Güvenlik ve intent analizi
            danger_result = danger_agent.generate_response(message)
            danger_status = "🟢" if danger_result == "safe" else "🔴"
            intent_result = analyze_prompt(message)
            intent_type = intent_result.get("intent", "unknown")

            agent_routing = f"""
### Agent Yönlendirme Bilgisi
{ "🟢" if danger_result == "safe" else "🔴" } Prompt is {danger_result.upper()} (DangerAgent)
🔀 Detected intent: {intent} (IntentAnalyzer)
🌐 Routed to {"RiskAnalyzerAgent" if intent == "risk" else f"{intent.capitalize()}Agent"}
"""

            cost_message = f"""
{agent_routing}
### Maliyet Bilgisi (Gemini 1.5 Flash Latest)
- *Prompt Token Sayısı:* {cost_info['prompt_tokens']}
- *Response Token Sayısı:* {cost_info['response_tokens']}
- *Toplam Token:* {cost_info['prompt_tokens'] + cost_info['response_tokens']}
- *Prompt Maliyeti:* ${cost_info['prompt_cost']:.6f}
- *Response Maliyeti:* ${cost_info['response_cost']:.6f}
- *Toplam Maliyet:* ${cost_info['total_cost']:.6f}
"""


            # JSON summary içeriğini asla eklemiyoruz
            formatted_response += f"\n\n📘 Summary:\n{llm_summary}"

            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": formatted_response})
            return history, gr.update(value=pdf_file, visible=True), cost_message

        else:
            formatted_response = str(response)
            cost_info = cost_calculator.calculate_cost(message, formatted_response, "gemini-1.5-flash-latest")
            cost_message = f"Toplam maliyet: ${cost_info['total_cost']:.6f}"
            return history + [{"role": "user", "content": message}, {"role": "assistant", "content": formatted_response}], gr.update(value=None, visible=False), cost_message

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Risk analyzer chatbot error: {e}\n{error_details}")
        return history + [{"role": "user", "content": message}, {"role": "assistant", "content": f"Bir hata oluştu: {str(e)}"}], gr.update(value=None, visible=False), "Maliyet hesaplanamadı"



# Create custom CSS for larger text with black and white theme
custom_css = """
:root {
    --primary-color: #000000;
    --secondary-color: #333333;
    --light-gray: #CCCCCC;
    --dark-gray: #222222;
    --text-color: #000000;
    --light-text: #FFFFFF;
    --accent-color: #555555;
    --border-radius: 8px;
    --box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
}

/* Login ve Register sayfası için özel stiller */
.auth-container {
    max-width: 400px !important;
    margin: 0 auto !important;
    padding: 2rem !important;
    background: white !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
}

.auth-container.dark {
    background: #1a1a1a !important;
}

.auth-title {
    text-align: center !important;
    font-size: 2rem !important;
    font-weight: bold !important;
    margin-bottom: 2rem !important;
    color: var(--primary-color) !important;
}

.auth-title.dark {
    color: var(--light-text) !important;
}

.auth-input {
    width: 100% !important;
    padding: 12px !important;
    margin-bottom: 1rem !important;
    border: 2px solid var(--light-gray) !important;
    border-radius: var(--border-radius) !important;
    font-size: 1rem !important;
    transition: all 0.3s ease !important;
}

.auth-input:focus {
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 0 2px rgba(0, 0, 0, 0.1) !important;
}

.auth-input.dark {
    background: #2a2a2a !important;
    border-color: #444 !important;
    color: white !important;
}

.auth-button {
    width: 100% !important;
    padding: 12px !important;
    background: var(--primary-color) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--border-radius) !important;
    font-size: 1rem !important;
    font-weight: bold !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    margin-top: 1rem !important;
}

.auth-button:hover {
    background: var(--secondary-color) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2) !important;
}

.auth-message {
    text-align: center !important;
    margin-top: 1rem !important;
    padding: 1rem !important;
    border-radius: var(--border-radius) !important;
    font-size: 0.9rem !important;
}

.auth-message.success {
    background: #e6ffe6 !important;
    color: #006600 !important;
    border: 1px solid #99ff99 !important;
}

.auth-message.error {
    background: #ffe6e6 !important;
    color: #cc0000 !important;
    border: 1px solid #ff9999 !important;
}

.auth-tabs {
    margin-bottom: 2rem !important;
}

.auth-tabs button {
    padding: 1rem 2rem !important;
    font-size: 1.1rem !important;
    font-weight: bold !important;
    border: none !important;
    background: none !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    color: #fff !important;
}

.auth-tabs button.selected {
    color: #fff !important;
    border-bottom: 3px solid #fff !important;
}

.auth-tabs.dark button {
    color: #fff !important;
}

.auth-tabs.dark button.selected {
    color: #fff !important;
    border-bottom-color: #fff !important;
}

.larger-text label {
    font-size: 22px !important;
    font-weight: bold !important;
    color: var(--primary-color) !important;
}

.larger-text .prose h1 {
    font-size: 36px !important;
    font-weight: bold !important;
    color: var(--primary-color) !important;
}

.larger-text .prose h2 {
    font-size: 32px !important;
    font-weight: bold !important;
    color: var(--primary-color) !important;
}

.larger-text .prose p {
    font-size: 20px !important;
    color: var(--text-color) !important;
}

.larger-text input, .larger-text select {
    font-size: 20px !important;
    padding: 12px !important;
    border-color: var(--secondary-color) !important;
}

.larger-text button {
    font-size: 20px !important;
    padding: 10px 20px !important;
    font-weight: bold !important;
    background-color: var(--primary-color) !important;
    color: var(--light-text) !important;
    border-radius: var(--border-radius) !important;
    transition: all 0.2s ease !important;
    margin: 5px !important;
}

.larger-text button:hover {
    background-color: var(--secondary-color) !important;
    transform: translateY(-2px) !important;
    box-shadow: var(--box-shadow) !important;
}

/* Override Gradio defaults for better visibility */
.dark .larger-text label, .dark .larger-text .prose h1, .dark .larger-text .prose h2 {
    color: var(--light-text) !important;
}

.dark .larger-text .prose p {
    color: #E0E0E0 !important;
}

.dark .larger-text input, .dark .larger-text select {
    background-color: var(--dark-gray) !important;
    color: white !important;
    border-color: var(--light-gray) !important;
}

/* Make dropdown options more readable */
.larger-text select option {
    font-size: 20px !important;
    padding: 10px !important;
}

/* Chatbox özel stilleri */
.large-chatbox {
    font-size: 18px !important;
    border-radius: 12px !important;
    box-shadow: var(--box-shadow) !important;
    margin-bottom: 20px !important;
    border: 1px solid var(--light-gray) !important;
}

.large-chatbox .message {
    padding: 15px !important;
    border-radius: 10px !important;
    margin: 8px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
}

/* Kullanıcı balonu */
.large-chatbox .user-message {
    background-color: #333333 !important;  /* koyu gri */
    color: #ffffff !important;             /* beyaz yazı */
    border-radius: 20px 20px 0 20px !important;
    max-width: 70% !important;
    align-self: flex-end !important;
    margin-left: 30% !important;
}

/* Bot balonu */
.large-chatbox .bot-message {
    background-color: #f0f0f0 !important;  /* açık gri */
    color: #000000 !important;             /* siyah yazı */
    border-radius: 20px 20px 20px 0 !important;
    max-width: 70% !important;
    align-self: flex-start !important;
    margin-right: 30% !important;
}

/* Chatbox dark mode */
.dark .large-chatbox {
    border-color: #444 !important;
}

.dark .large-chatbox .bot-message {
    background-color: var(--dark-gray) !important;
    color: var(--light-text) !important;
}

/* Belirgin input alanı */
.prominent-input {
    border: 2px solid var(--accent-color) !important;
    border-radius: var(--border-radius) !important;
    background-color: rgba(255, 255, 255, 0.05) !important;
    transition: all 0.3s ease !important;
    font-size: 18px !important;
    padding: 12px !important;
    box-shadow: var(--box-shadow) !important;
    margin-bottom: 10px !important;
}

.prominent-input:focus-within {
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 0 2px rgba(0, 0, 0, 0.2) !important;
}

.dark .prominent-input {
    background-color: rgba(0, 0, 0, 0.2) !important;
}

.dark .prominent-input:focus-within {
    box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.2) !important;
}

/* PDF yükleme alanı */
.pdf-upload-area {
    border: 2px dashed var(--accent-color) !important;
    border-radius: var(--border-radius) !important;
    padding: 15px !important;
    background-color: rgba(0, 0, 0, 0.02) !important;
    transition: all 0.3s ease !important;
    text-align: center !important;
    height: 100% !important;
    min-height: 150px !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    box-shadow: var(--box-shadow) !important;
}

.pdf-upload-area:hover {
    border-color: var(--primary-color) !important;
    background-color: rgba(0, 0, 0, 0.05) !important;
}

.dark .pdf-upload-area {
    background-color: rgba(255, 255, 255, 0.05) !important;
}

.dark .pdf-upload-area:hover {
    background-color: rgba(255, 255, 255, 0.1) !important;
}

/* Aksiyon butonları */
.action-button {
    background-color: var(--primary-color) !important;
    min-width: 120px !important;
}

.action-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2) !important;
}

.cost-panel {
    background-color: #f5f5f5 !important;
    padding: 20px !important;
    border-radius: 10px !important;
    border: 1px solid #ddd !important;
    margin: 10px !important;
    height: 100% !important;
    overflow-y: auto !important;
}

.cost-panel h3 {
    color: #333 !important;
    margin-bottom: 15px !important;
    font-size: 1.2em !important;
}

.cost-panel ul {
    list-style-type: none !important;
    padding: 0 !important;
    margin: 0 !important;
}

.cost-panel li {
    margin: 10px 0 !important;
    padding: 5px 0 !important;
    border-bottom: 1px solid #eee !important;
}

.dark .cost-panel {
    background-color: #2a2a2a !important;
    border-color: #444 !important;
}

.dark .cost-panel h3 {
    color: #fff !important;
}

.dark .cost-panel li {
    border-bottom-color: #444 !important;
}

.terms-inline {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 10px;
    font-size: 1rem;
}
.terms-inline input[type='checkbox'] {
    width: 18px;
    height: 18px;
    margin: 0 4px 0 0;
}
.terms-inline .accept-label {
    font-size: 1rem;
    font-weight: 400;
    color: #fff;
    margin: 0 2px 0 0;
}
.terms-inline .terms-link {
    font-size: 1rem;
    color: #4A90E2;
    text-decoration: underline;
    cursor: pointer;
    background: none;
    border: none;
    padding: 0;
    margin: 0 0 0 2px;
    font-weight: 400;
    transition: color 0.2s;
}
.terms-inline .terms-link:hover {
    color: #357ab8;
}
.gradio-container .component-value {
    display: none !important;
}
.component-value {
    display: none !important;
}

"""


# Create the main application
with gr.Blocks(title="FINSENTIO", css=custom_css, theme=gr.themes.Monochrome()) as app:
    with gr.Blocks(elem_classes=["larger-text"]):
        gr.Markdown("# FINSENTIO")
        
        # Create dashboard interface
        with gr.Group(visible=False) as dashboard:
            gr.Markdown("# Welcome to your Dashboard")
            
            with gr.Tabs():
                with gr.TabItem("Profile"):
                    gr.Markdown("## User Risk Profile Assessment")
                    gr.Markdown("Please answer the following questions to help us understand your investment preferences and risk tolerance.")
                    
                    # Question 1
                    with gr.Row():
                        with gr.Column():
                            risk_taker = gr.Radio(
                                label="1. How would your best friend describe you as a risk taker?",
                                choices=[
                                    "A real gambler",
                                    "Willing to take risks after completing adequate research",
                                    "Cautious",
                                    "A real risk avoider"
                                ],
                                value="Cautious",
                                scale=2
                            )
                    
                    # Question 2
                    with gr.Row():
                        with gr.Column():
                            risk_word = gr.Radio(
                                label="2. When you think of the word \"risk\", which of the following words comes to mind first?",
                                choices=[
                                    "Loss",
                                    "Uncertainty",
                                    "Opportunity",
                                    "Thrill"
                                ],
                                value="Uncertainty",
                                scale=2
                            )
                    
                    # Question 3
                    with gr.Row():
                        with gr.Column():
                            game_show = gr.Radio(
                                label="3. You are on a TV game show and can choose one of the following. Which would you take?",
                                choices=[
                                    "$1,000 in cash",
                                    "A 50% chance at winning $5,000",
                                    "A 25% chance at winning $10,000",
                                    "A 5% chance at winning $100,000"
                                ],
                                value="$1,000 in cash",
                                scale=2
                            )
                    
                    # Question 4
                    with gr.Row():
                        with gr.Column():
                            investment_allocation = gr.Radio(
                                label="4. If you had to invest $20,000, which of the following allocations would you find most appealing?",
                                choices=[
                                    "60% in low-risk, 30% in medium-risk, 10% in high-risk investments",
                                    "30% in low-risk, 30% in medium-risk, 40% in high-risk investments",
                                    "10% in low-risk, 40% in medium-risk, 50% in high-risk investments"
                                ],
                                value="60% in low-risk, 30% in medium-risk, 10% in high-risk investments",
                                scale=2
                            )
                    
                    # Question 5
                    with gr.Row():
                        with gr.Column():
                            market_follow = gr.Radio(
                                label="5. How often do you follow financial markets?",
                                choices=[
                                    "Daily",
                                    "Weekly",
                                    "Occasionally",
                                    "Never"
                                ],
                                value="Weekly",
                                scale=2
                            )
                    
                    # Question 6
                    with gr.Row():
                        with gr.Column():
                            new_investment = gr.Radio(
                                label="6. What would you do when you hear about a new investment opportunity?",
                                choices=[
                                    "Immediately jump in",
                                    "Research thoroughly before investing",
                                    "Ask others first, then decide",
                                    "Wait and observe over time"
                                ],
                                value="Research thoroughly before investing",
                                scale=2
                            )
                    
                    # Question 7
                    with gr.Row():
                        with gr.Column():
                            buy_things = gr.Radio(
                                label="7. I rarely buy things I don't need.",
                                choices=[
                                    "Agree",
                                    "Neutral",
                                    "Disagree"
                                ],
                                value="Neutral",
                                scale=2
                            )
                    
                    # Question 8
                    with gr.Row():
                        with gr.Column():
                            finance_reading = gr.Radio(
                                label="8. I like to read about finance and the economy.",
                                choices=[
                                    "Agree",
                                    "Neutral",
                                    "Disagree"
                                ],
                                value="Neutral",
                                scale=2
                            )
                    
                    # Question 9
                    with gr.Row():
                        with gr.Column():
                            previous_investments = gr.CheckboxGroup(
                                label="9. Which of the following have you invested in before? (You can choose more than one)",
                                choices=[
                                    "Stocks",
                                    "Cryptocurrency",
                                    "Foreign currencies",
                                    "Gold or other commodities",
                                    "Fixed deposit accounts",
                                    "I have never invested"
                                ],
                                value=[],
                                scale=2
                            )
                    
                    # Question 10
                    with gr.Row():
                        with gr.Column():
                            investment_goal = gr.Radio(
                                label="10. What is your main goal for investing?",
                                choices=[
                                    "Short-term profit",
                                    "Long-term savings",
                                    "Retirement planning",
                                    "Wealth preservation"
                                ],
                                value="Long-term savings",
                                scale=2
                            )
                    
                    # Save button
                    with gr.Row():
                        save_button = gr.Button("Save Profile", size="lg", variant="primary")
                    profile_message = gr.Markdown("")
                    
                    # Connect save button to function
                    save_button.click(
                        fn=save_profile,
                        inputs=[
                            risk_taker, risk_word, game_show, investment_allocation,
                            market_follow, new_investment, buy_things, finance_reading,
                            previous_investments, investment_goal
                        ],
                        outputs=profile_message
                    )
                
                with gr.TabItem("Education Chatbot"):
                    gr.Markdown("## Financial Education Chatbot")
                    gr.Markdown("Ask any questions about financial terms, concepts, or strategies to improve your knowledge!")
                    
                    with gr.Row():
                        # Sol taraf - Chat arayüzü
                        with gr.Column(scale=3):
                            # Education chatbot interface
                            education_chat = gr.Chatbot(
                                label="Chat History",
                                height=700,
                                value=[],
                                elem_classes=["large-chatbox"],
                                show_label=True,
                                container=True,
                                type="messages"
                            )
                            
                            # Kontrol paneli
                            with gr.Row(equal_height=True):
                                # Sol taraf - Soru girme alanı
                                with gr.Column(scale=4):
                                    education_msg = gr.Textbox(
                                        label="Your Question",
                                        placeholder="Ask me anything about finance...",
                                        container=True,
                                        lines=2,
                                        max_lines=4,
                                        elem_classes=["prominent-input"]
                                    )
                                    
                                    # Butonlar alt satırda
                                    with gr.Row():
                                        education_submit = gr.Button("Ask", variant="primary", size="lg", elem_classes=["action-button"])
                                        education_clear = gr.Button("Clear Chat", size="lg")
                                
                                # Sağ taraf - PDF yükleme
                                with gr.Column(scale=1, min_width=200):
                                    pdf_upload = gr.File(
                                        label="Upload PDF",
                                        file_types=[".pdf"],
                                        type="binary",
                                        elem_classes=["pdf-upload-area"]
                                    )
                        
                        # Sağ taraf - Maliyet bilgisi paneli
                        with gr.Column(scale=1):
                            cost_panel = gr.Markdown(
                                label="Maliyet Bilgisi",
                                value="### Maliyet Bilgisi\nHenüz bir soru sorulmadı.",
                                elem_classes=["cost-panel"]
                            )
                    
                    # Connect chatbot components
                    education_submit.click(
                        fn=education_chatbot,
                        inputs=[education_msg, pdf_upload, education_chat],
                        outputs=[education_chat, cost_panel],
                        api_name="education_chat"
                    )
                    education_clear.click(
                        lambda: [[], "### Maliyet Bilgisi\nHenüz bir soru sorulmadı."],
                        None,
                        [education_chat, cost_panel]
                    )
                
                with gr.TabItem("Financial Adviser & Analyzer Chatbot"):
                    gr.Markdown("## Financial Adviser & Analyzer Chatbot")
                    gr.Markdown("Get personalized financial advice and analysis based on your profile and market conditions.")
                    
                    with gr.Row():
                        # Sol taraf - Chat arayüzü
                        with gr.Column(scale=3):
                            # Adviser chatbot interface
                            adviser_chat = gr.Chatbot(
                                label="Chat History",
                                height=700,
                                value=[],
                                type="messages"
                            )
                            adviser_msg = gr.Textbox(
                                label="Your Question",
                                placeholder="Ask for financial advice or market analysis...",
                                scale=7
                            )
                            with gr.Row():
                                adviser_submit = gr.Button("Ask", scale=1, variant="primary")
                                adviser_clear = gr.Button("Clear Chat", scale=1)
                            
                            # Excel download link
                            excel_link = gr.File(
                                label="Download Excel Report",
                                visible=False,
                                interactive=True,
                                type="filepath"
                            )
                        
                        # Sağ taraf - Maliyet bilgisi paneli
                        with gr.Column(scale=1):
                            adviser_cost_panel = gr.Markdown(
                                label="Maliyet Bilgisi",
                                value="### Maliyet Bilgisi\nHenüz bir soru sorulmadı.",
                                elem_classes=["cost-panel"]
                            )
                    
                    # Connect chatbot components
                    outputs = adviser_submit.click(
                        fn=advisor_chatbot,
                        inputs=[adviser_msg, adviser_chat],
                        outputs=[adviser_chat, excel_link, adviser_cost_panel]
                    )
                    adviser_clear.click(
                        lambda: [[], None, "### Maliyet Bilgisi\nHenüz bir soru sorulmadı."],
                        None,
                        [adviser_chat, excel_link, adviser_cost_panel]
                    )
            
            # Logout button
            with gr.Row():
                logout_button = gr.Button("Logout", size="lg", variant="stop")
        
        # Create auth interface
        with gr.Group(visible=True) as auth_interface:
            with gr.Group(elem_classes=["auth-container"]):
                with gr.Tabs(elem_classes=["auth-tabs"]) as auth_tabs:
                    with gr.TabItem("Login", elem_classes=["auth-tab"]):
                        gr.Markdown("## Login to Your Account", elem_classes=["auth-title"])
                        username_login = gr.Textbox(
                            label="Username",
                            placeholder="Enter your username",
                            elem_classes=["auth-input"]
                        )
                        password_login = gr.Textbox(
                            label="Password",
                            type="password",
                            placeholder="Enter your password",
                            elem_classes=["auth-input"]
                        )
                        # Accept terms and conditions: Centered, single HTML block above login button
                        gr.HTML(
                            '''
                            <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 12px; margin-top: 24px;">
                                <input type="checkbox" id="accept-terms-html" style="width:18px;height:18px;margin-right:8px;">
                                <span style="font-size:1rem;color:#fff;">Accept </span>
                                <a href="javascript:void(0);" id="terms-link-html" style="color:#4A90E2;text-decoration:underline;cursor:pointer;font-size:1rem;margin-left:4px;">Terms and Conditions</a>
                            </div>
                            <script>
                            document.addEventListener('DOMContentLoaded', function() {
                                var link = document.getElementById('terms-link-html');
                                if(link) link.onclick = function() {
                                    document.querySelector('[id^=\'terms-btn\']').click();
                                };
                            });
                            </script>
                            '''
                        )
                        # Modal benzeri popup için Markdown ve kapat butonu
                        terms_popup = gr.Markdown(
                            value="""# Terms and Conditions\n\nHere you can put your terms and conditions text.\n\nLorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla accumsan, metus ultrices eleifend gravida...\n\n- You must accept these terms to use the service.\n- Your data is handled securely.\n- ...\n\n""",
                            visible=False,
                            elem_id="terms-popup-md",
                            elem_classes=["cost-panel"]
                        )
                        terms_btn = gr.Button(
                            value="Terms and Conditions",
                            elem_id="terms-btn",
                            size="sm",
                            variant="secondary",
                            visible=False  # Only for JS trigger
                        )
                        close_terms_btn = gr.Button(
                            value="Close",
                            visible=False,
                            elem_id="close-terms-btn",
                            size="sm"
                        )
                        # Checkbox işaretlenince login_button aktif olsun (Blocks context içinde olmalı)
                        with gr.Row():
                            login_button = gr.Button(
                                "Login",
                                size="lg",
                                variant="primary",
                                elem_classes=["auth-button"],
                                interactive=True  # Başlangıçta pasif
                            )
                        login_message = gr.Markdown(
                            elem_classes=["auth-message"]
                        )
                    
                    with gr.TabItem("Register", elem_classes=["auth-tab"]):
                        gr.Markdown("## Create New Account", elem_classes=["auth-title"])
                        username_register = gr.Textbox(
                            label="Username",
                            placeholder="Choose a username",
                            elem_classes=["auth-input"]
                        )
                        email_register = gr.Textbox(
                            label="Email",
                            placeholder="Enter your email",
                            elem_classes=["auth-input"]
                        )
                        password_register = gr.Textbox(
                            label="Password",
                            type="password",
                            placeholder="Choose a password",
                            elem_classes=["auth-input"]
                        )
                        confirm_password = gr.Textbox(
                            label="Confirm Password",
                            type="password",
                            placeholder="Confirm your password",
                            elem_classes=["auth-input"]
                        )
                        with gr.Row():
                            register_button = gr.Button(
                                "Register",
                                size="lg",
                                variant="primary",
                                elem_classes=["auth-button"]
                            )
                        register_message = gr.Markdown(
                            elem_classes=["auth-message"]
                        )
        
        # Connect the buttons to functions
        login_button.click(
            fn=login,
            inputs=[username_login, password_login],
            outputs=[login_message, auth_interface, dashboard]
        ).then(
            fn=load_profile,
            inputs=None,
            outputs=[
                risk_taker, risk_word, game_show, investment_allocation,
                market_follow, new_investment, buy_things, finance_reading,
                previous_investments, investment_goal
            ]
        )
        
        register_button.click(
            fn=register,
            inputs=[username_register, password_register, email_register, confirm_password],
            outputs=[register_message, auth_interface, dashboard]
        )
        
        logout_button.click(
            fn=logout,
            outputs=[auth_interface, dashboard]
        )

if __name__ == "__main__":
    app.launch(allowed_paths=["C:/Users/user/Desktop/deneme_sonllm"])