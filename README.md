
Of course! This is an excellent and well-thought-out plan. My role here is to act as your co-pilot, guiding you through the implementation of your own architecture, step-by-step. Let's build this together.

We'll follow your structure precisely. You've already done the hard work of designing the system; now, let's bring it to life.

---

### **Preparation: Your Google Cloud Environment**

Before we start, make sure you have a Google Cloud Project with billing enabled. Then, enable the necessary APIs. This is a crucial first step that prevents errors later.

1.  **Go to the Google Cloud Console.**
2.  Make sure your project is selected.
3.  Use the search bar or the navigation menu to go to "APIs & Services" -> "Library".
4.  Search for and **Enable** each of the following APIs:
    *   `Dialogflow API`
    *   `Cloud Functions API`
    *   `Vertex AI API`
    *   `Cloud Speech-to-Text API`
    *   `Cloud Text-to-Speech API`
    *   `Cloud Build API` (often enabled by default, but needed for Cloud Functions)
    *   (Optional but Recommended) `Firestore API`

Now, let's proceed with your plan.

---

### **1. System Architecture**

Your Mermaid diagram and component breakdown are perfect. They serve as our blueprint for the entire project. We will refer back to this as we build each piece.

---

### **2. Step-by-Step: Setting up Dialogflow CX for Voice Calls**

Let's configure the "brain" of our voice assistant.

#### **A. Create a Dialogflow CX Agent**

1.  Navigate to the [Dialogflow CX Console](https://dialogflow.cloud.google.com/cx/).
2.  Click **+ Create agent**.
3.  **Agent Name:** Enter `AgriAssistant`.
4.  **Location:** Select a region like `us-central1`. It's important to use the same region for your Cloud Function later.
5.  **Default Language:** Select **Hindi (hi)** from the dropdown.
6.  Click the **Languages** tab.
7.  Click **+ Add Language** and add **Marathi (mr)**.
8.  Add **English (en)** as well. This is very useful for your own testing and development.
9.  Click **Create**.

#### **B. Configure Agent Settings for Voice & Language**

This is the most critical part for making the voice experience natural and multilingual.

1.  Inside your new agent, click on **Agent Settings** (the gear icon ⚙️ on the left menu).
2.  Select the **Speech and IVR** tab.
3.  **Text-to-Speech:**
    *   **Voice selection:** You need to set a voice for each language.
        *   For `hi` (Hindi): Click **Change voice** and choose a high-quality voice like `hi-IN-Wavenet-D` (male) or `hi-IN-Wavenet-C` (female).
        *   For `mr` (Marathi): Click the language tab, select Marathi, click **Change voice**, and choose `mr-IN-Wavenet-A` (female) or `mr-IN-Wavenet-B` (male).
        *   Do the same for English (e.g., `en-IN-Wavenet-A`).
4.  **Speech-to-Text:**
    *   **Enable Auto Language Detection:** **Check this box. This is the key to multilingual understanding.**
    *   **Languages:** Click into the languages field and add `hi-IN` (Hindi - India), `mr-IN` (Marathi - India), and `en-IN` (English - India). The agent will now try to recognize speech from any of these languages automatically.
    *   **Model:** Choose **telephony** for the best results over a phone call.
5.  **Advanced Speech Settings:**
    *   **Enable Barge-in:** Check this box. It allows a farmer to interrupt the assistant, making the conversation more natural.
    *   **No Speech Timeout:** Set this to `5s`. If the farmer doesn't say anything for 5 seconds, the agent will repeat the prompt or move on.
6.  Click **Save**.

#### **C. Design the Basic Flow (Default Start Flow)**

Now we design the conversation.

1.  Click the **Build** tab on the left. You'll see the **Default Start Flow** with a **Start** page. Click on the **Start** page node.

2.  **Initial Welcome & Language Choice:**
    *   In the panel that opens, find the **Entry fulfillment** section and click **Edit fulfillment**.
    *   Under **Agent says**, we'll add language-specific greetings.
        *   Click the text box, then click **+ Add language variant**.
        *   **For `hi`:** "नमस्ते! मैं आपका कृषि सहायक हूँ। आप मुझसे खेती सम्बंधित सवाल पूछ सकते हैं, या मौसम की जानकारी ले सकते हैं। आप किस भाषा में बात करना पसंद करेंगे? हिंदी या मराठी?"
        *   **For `mr`:** "नमस्कार! मी तुमचा कृषी सहाय्यक आहे. तुम्ही मला शेती संबंधित प्रश्न विचारू शकता, किंवा हवामानाची माहिती घेऊ शकता. तुम्हाला कोणत्या भाषेत बोलायला आवडेल? हिंदी की मराठी?"
        *   **For `en`:** "Hello! I'm your Agri Assistant. Which language would you prefer? Hindi or Marathi?"
    *   Click **Save**.

3.  **Create Intents for Language Selection:**
    *   From the left menu, click **Manage** -> **Intents**.
    *   Click **+ Create**. Name it `select_hindi`. Add training phrases like "Hindi", "हिंदी", "हिंदी में", "हिंदी में बोलो". Click **Save**.
    *   Click **+ Create** again. Name it `select_marathi`. Add training phrases like "Marathi", "मराठी", "मराठीत", "मराठीत बोला". Click **Save**.

4.  **Create the `MainInteractionPage`:**
    *   Back in the **Build** tab, click the **+** button in the "Pages" list on the left.
    *   Name the new page `MainInteractionPage`.

5.  **Route from Start to `MainInteractionPage`:**
    *   Go back to the **Start** page.
    *   Under the **Routes** section, click **+ New Route**.
        *   **Intent:** Select `select_hindi`.
        *   **Parameter Presets:** Under `Fulfillment`, click **+ Add**, select **Parameter preset**, set `session.params.selected_language` to `hi`.
        *   **Transition:** Select Page -> `MainInteractionPage`.
    *   Click **+ New Route** again.
        *   **Intent:** Select `select_marathi`.
        *   **Parameter Presets:** Under `Fulfillment`, set `session.params.selected_language` to `mr`.
        *   **Transition:** Select Page -> `MainInteractionPage`.
    *   Click **Save**.

6.  **Configure `MainInteractionPage`:**
    *   Click on the `MainInteractionPage` node.
    *   **Entry Fulfillment:** Add a conditional response.
        *   **Condition 1:** `if $session.params.selected_language = "hi"`
        *   **Agent says:** "आप क्या पूछना चाहते हैं?"
        *   **Condition 2:** `if $session.params.selected_language = "mr"`
        *   **Agent says:** "तुम्हाला काय विचारायचे आहे?"
        *   **Default:** "How can I help you?"
    *   **Create Intents for Farmer Queries:**
        *   Go to **Manage** -> **Intents**.
        *   **`AskGeneralQuestion`**: Create this intent. Add many examples in both languages:
            *   "गेंहू की बुवाई कब करें"
            *   "टमाटर में कौन सी खाद डालें"
            *   "माझ्या पिकाला कीड लागली आहे"
            *   "ऊस कसा लावावा"
            *   "my crop is diseased"
        *   Inside this intent, create a parameter named `user_query` of type `@sys.any` to capture the entire question.
        *   **`AskWeather`**: Create this intent. Add training phrases:
            *   "आज मौसम कैसा है"
            *   "बारिश कब होगी"
            *   "हवामान कसे आहे"
            *   "पाऊस कधी येणार"
            *   "weather today"
        *   Inside this intent, create a parameter `location` of type `@sys.location`.
    *   **Add Routes to `MainInteractionPage`:**
        *   **Route 1 (General Query):**
            *   **Intent:** `AskGeneralQuestion`.
            *   **Fulfillment:** We will add the Webhook here in a later step.
            *   **Transition:** `MainInteractionPage` (to stay and ask another question).
        *   **Route 2 (Weather Query):**
            *   **Intent:** `AskWeather`.
            *   **Fulfillment:** We will add the Webhook here too.
            *   **Transition:** `MainInteractionPage`.

#### **D. Set up Phone Gateway Integration**

1.  Go to the **Manage** tab on the left.
2.  Click **Integrations**.
3.  Click **Connect** on **Dialogflow CX Phone Gateway**.
4.  Click **+ Create new**.
5.  Select the country (e.g., India +91).
6.  A phone number will be assigned to you. Click **Save**.

**You now have a phone number that connects directly to your Dialogflow agent!**

---

### **3. Guide: Using Gemini via Webhook**

Now, let's create the backend "power" for our assistant.

#### **A. Create a Cloud Function**

1.  Go to the [Google Cloud Functions Console](https://console.cloud.google.com/functions).
2.  Click **CREATE FUNCTION**.
3.  **Basics:**
    *   **Environment:** **2nd gen**.
    *   **Function Name:** `agri-assistant-webhook`.
    *   **Region:** Select the same region as your Dialogflow agent (e.g., `us-central1`).
4.  **Trigger:**
    *   **Trigger type:** HTTP.
    *   **Authentication:** Select **Allow unauthenticated invocations**. (This is for simplicity. In production, you would secure this).
5.  **Runtime, build and connections settings:**
    *   **Runtime service account:** Note the service account email. We'll need to give it permissions.
    *   **Runtime environment variables:** Click **ADD VARIABLE**. We will add `OPENWEATHER_API_KEY` here later.
6.  Click **Next**.

#### **B. `main.py` for the Cloud Function**

*   **Runtime:** Select **Python 3.10** (or newer).
*   **Source code:** Select **Inline Editor**.
*   **Entry point:** `handle_webhook`.
*   Paste the following code into the `main.py` editor.

```python
import functions_framework
import json
import os
import requests
from vertexai.preview.generative_models import GenerativeModel

# --- Configuration ---
PROJECT_ID = os.environ.get("GCP_PROJECT", "your-gcp-project-id")
LOCATION = "us-central1"  # Your GCP region for Vertex AI
GEMINI_MODEL_NAME = "gemini-1.0-pro"

# Get API key from environment variables. Set this in the Cloud Function settings.
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "YOUR_DEFAULT_KEY")
OPENWEATHER_API_URL = "http://api.openweathermap.org/data/2.5/weather"


def get_gemini_response(query: str, language_code: str) -> str:
    """Generates a response from the Gemini model."""
    try:
        model = GenerativeModel(GEMINI_MODEL_NAME)
        
        language_instruction = "Please answer in simple, friendly English."
        error_message = "I'm sorry, I couldn't find an answer for that right now."
        tech_error_message = "I'm having a technical issue with my knowledge base."

        if language_code.startswith("hi"):
            language_instruction = "Please answer in simple, friendly Hindi. Keep sentences short."
            error_message = "मुझे क्षमा करें, मैं अभी इसका उत्तर नहीं दे पा रहा हूँ।"
            tech_error_message = "तकनीकी समस्या के कारण मैं अभी जवाब नहीं दे पा रहा हूँ।"
        elif language_code.startswith("mr"):
            language_instruction = "Please answer in simple, friendly Marathi. Keep sentences short."
            error_message = "मला माफ करा, मी सध्या याचे उत्तर देऊ शकत नाही."
            tech_error_message = "तांत्रिक समस्येमुळे मी सध्या प्रतिसाद देऊ शकत नाही."

        prompt = f"""
        You are a helpful and friendly agricultural assistant for Indian farmers.
        Your goal is to provide clear, concise, and actionable advice.
        {language_instruction}
        Do not use very complex or technical terms. If you must, explain them simply.
        Farmer's query: "{query}"

        Your answer:
        """

        response = model.generate_content(prompt)

        if response.candidates and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text
        else:
            print(f"Warning: Gemini response was empty for query: {query}")
            return error_message
    except Exception as e:
        print(f"Error calling Gemini: {e}")
        return tech_error_message

def get_weather_info(location_query: str, language_code: str) -> str:
    """Fetches and formats weather information from OpenWeatherMap."""
    lang_param = language_code.split('-')[0] # 'hi', 'mr', 'en'
    
    # Error messages in different languages
    error_messages = {
        'hi': f"{location_query} के लिए मौसम की जानकारी प्राप्त करने में असमर्थ।",
        'mr': f"{location_query} साठी हवामान माहिती मिळवता आली नाही.",
        'en': f"Sorry, I couldn't get weather information for {location_query}."
    }
    service_error_messages = {
        'hi': "मौसम सेवा से संपर्क करने में समस्या आ रही है।",
        'mr': "हवामान सेवेशी संपर्क साधण्यात समस्या येत आहे.",
        'en': "I'm having trouble connecting to the weather service."
    }

    params = {
        'q': location_query,
        'appid': OPENWEATHER_API_KEY,
        'units': 'metric',  # For Celsius
        'lang': lang_param
    }
    try:
        response = requests.get(OPENWEATHER_API_URL, params=params)
        response.raise_for_status()
        data = response.json()

        if data.get("cod") != 200:
            return error_messages.get(lang_param, error_messages['en'])
        
        city_name = data.get("name", location_query)
        description = data['weather'][0]['description']
        temp = data['main']['temp']
        humidity = data['main']['humidity']
        
        if lang_param == "hi":
            return f"{city_name} में अभी मौसम: {description}. तापमान {temp}°C और नमी {humidity}% है।"
        elif lang_param == "mr":
            return f"{city_name} मधील सध्याचे हवामान: {description}. तापमान {temp}°C आणि आर्द्रता {humidity}% आहे."
        else:
            return f"In {city_name}, the weather is {description} with a temperature of {temp}°C and humidity at {humidity}%."

    except requests.exceptions.RequestException as e:
        print(f"Error calling OpenWeatherMap API: {e}")
        return service_error_messages.get(lang_param, service_error_messages['en'])


@functions_framework.http
def handle_webhook(request):
    """Handles webhook requests from Dialogflow CX."""
    request_json = request.get_json(silent=True)
    tag = request_json.get('fulfillmentInfo', {}).get('tag')
    
    # Determine the language to use for the response
    session_params = request_json.get('sessionInfo', {}).get('parameters', {})
    lang_code = session_params.get('selected_language')
    if not lang_code:
        lang_code = request_json.get('languageCode', 'hi-IN') # Default to Hindi

    final_response_text = ""

    if tag == 'agri_query_gemini':
        user_query = session_params.get('user_query', '')
        if not user_query:
            user_query = request_json.get('text', '') # Fallback to raw text
        
        if user_query:
            final_response_text = get_gemini_response(user_query, lang_code)
        else: # Should not happen if intent is matched, but as a safeguard
            final_response_text = "आपने क्या पूछा, मैं समझ नहीं पाया।"
            if lang_code.startswith("mr"):
                final_response_text = "तुम्ही काय विचारले ते मला समजले नाही."

    elif tag == 'weather_query_api':
        location_param = session_params.get('location', '')
        location_name = ""
        if isinstance(location_param, dict):
            location_name = location_param.get('city', '')
        elif isinstance(location_param, str):
            location_name = location_param
            
        if not location_name:
            # This is a signal to Dialogflow to ask for the required 'location' parameter
            # This part should be handled by making the 'location' parameter required in Dialogflow
            # So, we return an empty response, and Dialogflow will use its own reprompt.
            pass
        else:
            final_response_text = get_weather_info(location_name, lang_code)
            
    else:
        final_response_text = "मैं इस अनुरोध को समझ नहीं सका।"
        if lang_code.startswith("mr"):
            final_response_text = "मला ही विनंती समजली नाही."
    
    # Construct the final JSON response for Dialogflow CX
    response_json = {
        "fulfillment_response": {
            "messages": [{"text": {"text": [final_response_text]}}]
        }
    }
    
    return json.dumps(response_json)
```

#### **C. `requirements.txt` for the Cloud Function**

*   In the Inline Editor, switch to the `requirements.txt` file.
*   Paste the following:

```txt
functions-framework>=3.0.0
google-cloud-aiplatform>=1.47.0
requests>=2.25.0
vertexai>=1.47.0
```

#### **D. Deploy the Cloud Function**

1.  **Grant Permissions:**
    *   Go to the [IAM & Admin page](https://console.cloud.google.com/iam-admin/iam).
    *   Find the service account associated with your function (it looks like `...-compute@developer.gserviceaccount.com` or similar).
    *   Click the pencil icon to edit its roles.
    *   Click **+ ADD ANOTHER ROLE** and add the **Vertex AI User** role. This allows the function to call the Gemini API.
    *   Click **Save**.
2.  **Add Environment Variable:**
    *   Get a free API key from [OpenWeatherMap](https://home.openweathermap.org/api_keys).
    *   Go back to your Cloud Function's configuration page.
    *   Under **Runtime, build and connections settings** -> **Runtime environment variables**, add a variable:
        *   **Name:** `OPENWEATHER_API_KEY`
        *   **Value:** `Your_Actual_OpenWeatherMap_API_Key`
3.  Click **DEPLOY**. It will take a few minutes. Once done, go to the **TRIGGER** tab and copy the **HTTPS Endpoint URL**.

#### **E. Connect Webhook in Dialogflow CX**

1.  In your Dialogflow agent, go to the **Manage** tab -> **Webhooks**.
2.  Click **+ Create**.
3.  **Name:** `GeminiAgriWebhook`.
4.  **Webhook URL:** Paste the Trigger URL you copied from your Cloud Function.
5.  **Timeout:** Set to `10` seconds (to give Gemini time to respond).
6.  Click **Save**.

#### **F. Use Webhook in your Flow**

1.  Go back to the **Build** tab and click on `MainInteractionPage`.
2.  Find the route for the `AskGeneralQuestion` intent.
3.  In the **Fulfillment** section, click **+ Add**.
4.  Select **Webhook** and choose your `GeminiAgriWebhook`.
5.  A **Tag** field will appear. Enter `agri_query_gemini`. This tag is how our Python code knows what to do.
6.  Now, find the route for the `AskWeather` intent.
7.  In its **Fulfillment**, select the same `GeminiAgriWebhook`.
8.  For its **Tag**, enter `weather_query_api`.
9.  Click **Save**.

---

### **4. Connect Weather Alerts**

This is mostly done! The final piece is ensuring Dialogflow asks for the location if it's missing.

1.  In your Dialogflow agent, go to the **Build** tab and select `MainInteractionPage`.
2.  Click on the `AskWeather` intent route.
3.  The `location` parameter should be listed. If not, add it.
4.  **Check the "Required" box** for the `location` parameter.
5.  This will reveal a new fulfillment section specifically for when this parameter is missing. Click **Edit fulfillment**.
6.  Under **Agent says**, add language-specific prompts:
    *   **`hi`:** "ज़रूर, आप किस जगह का मौसम जानना चाहते हैं?"
    *   **`mr`:** "नक्कीच, तुम्हाला कोणत्या ठिकाणचे हवामान जाणून घ्यायचे आहे?"
    *   **`en`:** "Sure, for which location would you like the weather?"
7.  Click **Save**.

Now, if a farmer says "What's the weather like?", the agent will automatically reply "Sure, for which location...?"

---

### **5. Deploy and Test the Full Pipeline**

You are ready to test the end-to-end system!

1.  **Test with Dialogflow Simulator:**
    *   Click **Test Agent** in the top-right of the Dialogflow console.
    *   Try different queries in both Hindi and Marathi.
        *   "नमस्ते" -> "मराठी" -> "मला ऊसाच्या लागवडीबद्दल सांगा" (Hello -> Marathi -> Tell me about sugarcane cultivation)
        *   "हवामान कसे आहे?" -> "पुणे" (How is the weather? -> Pune)
        *   "गेहूं के लिए सबसे अच्छी खाद कौन सी है?" (Which is the best fertilizer for wheat?)
    *   Watch the conversation flow and check the responses.

2.  **Test with a Real Phone Call:**
    *   Find the phone number from the **Manage** -> **Integrations** -> **Dialogflow CX Phone Gateway** page.
    *   **Call the number.**
    *   Speak naturally and test the same scenarios. Listen to the voice quality and response speed.

3.  **Debugging:**
    *   **Dialogflow:** In the simulator, you can see the matched intent and extracted parameters for each turn.
    *   **Cloud Function:** Go to the [Cloud Logging page](https://console.cloud.google.com/logs/viewer) in the Google Cloud Console. You can filter for your function (`agri-assistant-webhook`) to see any `print()` statements or error messages.

You have now successfully built and deployed a voice-based agricultural assistant using the powerful combination of Dialogflow CX and Gemini. Congratulations
