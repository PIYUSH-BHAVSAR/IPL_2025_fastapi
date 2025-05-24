Okay, this is an exciting and impactful project! Let's architect and build this Voice-Based Agri Assistant step-by-step.

**Core Principles:**

*   **Simplicity for Farmers:** The interaction must be natural and easy.
*   **Modularity:** Each component should be distinct and manageable.
*   **Scalability:** Leverage GCP's managed services.
*   **Cost-Effectiveness:** Choose appropriate service tiers.

---

## 1. System Architecture

Here's a high-level diagram of the proposed system:

```mermaid
graph TD
    A[Farmer on Phone] -- PSTN Call --> B(Dialogflow CX Phone Gateway)
    B -- Audio Stream --> C{Dialogflow CX Agent}
    C -- Speech-to-Text (Google STT) --> D[Farmer's Utterance in Text]
    C -- Language Detection --> L_DETECTED[Detected Language: Hindi/Marathi/etc.]

    subgraph Dialogflow CX Core Logic
        direction LR
        D -- Query & Language --> E{Intent Matching & Flow Control}
        E -- Parameter Extraction --> PARAMS[Extracted Parameters e.g., Crop, Location]
    end

    E -- "General Query" Route --> F[Webhook Call: AgriQueryFunction]
    E -- "Weather Query" Route --> G[Webhook Call: WeatherFunction OR same AgriQueryFunction]
    E -- "Change Language" Route --> H[Set Session Parameter: Language]

    subgraph Cloud Functions Backend
        direction TB
        F --> I{Cloud Function: AgriQueryHandler}
        I -- Query & Language --> J[Vertex AI Gemini API]
        J -- Gemini Response --> I
        I -- Formatted Text Response --> C

        G --> I  // Can be the same function
        I -- Location & Language --> K[OpenWeatherMap API / Google Weather API]
        K -- Weather Data --> I
    end

    C -- Text Response & Language --> M[Text-to-Speech (Google TTS)]
    M -- Synthesized Audio --> B
    B -- Audio Response --> A

    subgraph Optional Components
        C -- Interaction Data --> N[Firestore (Logging)]
        L_DETECTED -- If needed --> O[Google Maps API (Location to Language Fallback - if caller ID provides usable location)]
    end

    style A fill:#lightgrey,stroke:#333,stroke-width:2px
    style B fill:#f9d,stroke:#333,stroke-width:2px
    style C fill:#f9d,stroke:#333,stroke-width:2px
    style I fill:#ccf,stroke:#333,stroke-width:2px
    style J fill:#bbf,stroke:#333,stroke-width:2px
    style K fill:#bbf,stroke:#333,stroke-width:2px
    style N fill:#e6ffe6,stroke:#333,stroke-width:2px
    style O fill:#e6ffe6,stroke:#333,stroke-width:2px
```

**Component Breakdown:**

1.  **Farmer:** Initiates a call.
2.  **Dialogflow CX Phone Gateway:** Handles the telephony integration. You'll get a phone number that farmers can call.
3.  **Dialogflow CX Agent:**
    *   **Speech-to-Text (STT):** Converts farmer's speech to text. Crucially, configure this for multi-language detection.
    *   **Language Detection:** Dialogflow can attempt to auto-detect or you can explicitly ask.
    *   **Intent Matching & Flow Control:** Determines what the farmer wants (general query, weather, change language).
    *   **Text-to-Speech (TTS):** Converts the assistant's text response back to speech in the appropriate language and voice.
4.  **Cloud Function (Webhook - `AgriQueryHandler`):**
    *   Receives requests from Dialogflow CX.
    *   Interfaces with Vertex AI Gemini for generating answers.
    *   Interfaces with OpenWeatherMap API (or Google Weather) for weather data.
    *   Formats responses to be simple and friendly.
5.  **Vertex AI Gemini API:** The LLM that generates intelligent answers to agricultural queries.
6.  **OpenWeatherMap API / Google Weather API:** Provides weather information.
7.  **Firestore:** (Optional but recommended) For logging interactions, user preferences (like chosen language).
8.  **Google Maps API:** (Optional) If you want to attempt location-based language fallback from caller ID (this is complex and may have privacy implications, direct detection/prompting is better).

---

## 2. Step-by-Step: Setting up Dialogflow CX for Voice Calls

**Pre-requisites:**

*   Google Cloud Project set up with billing enabled.
*   APIs enabled: Dialogflow API, Cloud Functions API, Vertex AI API, Speech-to-Text API, Text-to-Speech API, (Optional: Firestore API, Maps APIs).

**A. Create a Dialogflow CX Agent:**

1.  Go to the Dialogflow CX console.
2.  Click "+ Create agent".
3.  Name it (e.g., `AgriAssistant`).
4.  Select your GCP project.
5.  Choose your region (e.g., `us-central1`).
6.  **Default Language:** Select `Hindi (hi)`.
7.  **Additional Languages:** Add `Marathi (mr)` and `English (en)` (English is good for testing/development).
8.  Click "Create".

**B. Configure Agent Settings for Voice & Language:**

1.  Go to "Agent Settings" (gear icon).
2.  **Speech and IVR Tab:**
    *   **Enable Auto Speech Adaptation:** Recommended.
    *   **Text-to-Speech > Voice selection:**
        *   For `hi` (Hindi): Choose a suitable voice (e.g., `hi-IN-Wavenet-D` for a good male voice, or `hi-IN-Wavenet-A` for female).
        *   For `mr` (Marathi): Choose `mr-IN-Wavenet-A` (female) or `mr-IN-Wavenet-B` (male).
        *   Configure for English as well.
    *   **Speech-to-Text > Enable Auto Language Detection:** **CRITICAL!**
        *   Add `hi-IN`, `mr-IN`, `en-US` to the list of languages for detection. Dialogflow will try to detect among these.
        *   Set "Model" to "Latest Long" or "Telephony" for best voice recognition over phone.
    *   **Advanced Speech Settings:**
        *   **No Speech Timeout:** e.g., 5 seconds.
        *   **Enable Barge-in:** Allow users to interrupt.
        *   **DTMF Settings:** (Optional for now, but useful for menu-driven fallbacks).
3.  **Languages Tab:** Confirm your languages are listed. You can set the default language for the agent here too.

**C. Design the Basic Flow (Default Start Flow):**

1.  **Start Page:**
    *   **Entry Fulfillment:**
        *   Add a "Fulfill" response.
        *   Agent says:
            *   Hindi: "नमस्ते! मैं आपका कृषि सहायक हूँ। आप मुझसे खेती सम्बंधित सवाल पूछ सकते हैं, या मौसम की जानकारी ले सकते हैं। आप किस भाषा में बात करना पसंद करेंगे? हिंदी या मराठी?"
            *   Marathi: "नमस्कार! मी तुमचा कृषी सहाय्यक आहे. तुम्ही मला शेती संबंधित प्रश्न विचारू शकता, किंवा हवामानाची माहिती घेऊ शकता. तुम्हाला कोणत्या भाषेत बोलायला आवडेल? हिंदी की मराठी?"
            *   English: "Hello! I'm your Agri Assistant. You can ask me farming questions or get weather updates. Which language would you prefer? Hindi or Marathi?"
        *   This initial prompt helps set the language explicitly if auto-detection isn't perfect.
    *   **Parameters:**
        *   Create a session parameter: `session.params.selected_language` (Text type).
    *   **Routes (for language selection):**
        *   **Route 1 (Hindi Selected):**
            *   Intent: Create a new intent `select_hindi`. Training phrases: "Hindi", "हिंदी", "हिंदी में".
            *   Condition: `$intent.name = "select_hindi"`
            *   Parameter Preset: Set `session.params.selected_language` to `hi`.
            *   Transition: Go to a new Page, e.g., `MainInteractionPage`.
        *   **Route 2 (Marathi Selected):**
            *   Intent: Create a new intent `select_marathi`. Training phrases: "Marathi", "मराठी", "मराठीत".
            *   Condition: `$intent.name = "select_marathi"`
            *   Parameter Preset: Set `session.params.selected_language` to `mr`.
            *   Transition: Go to `MainInteractionPage`.
        *   **Route 3 (Language Auto-Detected or Fallback):**
            *   Condition: `true` (as a fallback)
            *   Transition: Go to `MainInteractionPage`. (Here, you'd rely on `$request.language_code` or STT auto-detection).

2.  **`MainInteractionPage`:**
    *   **Entry Fulfillment:**
        *   Agent says (dynamic based on language, you'll set this up using conditions or webhook later):
            *   If `session.params.selected_language = "hi"`: "आप क्या पूछना चाहते हैं?"
            *   If `session.params.selected_language = "mr"`: "तुम्हाला काय विचारायचे आहे?"
            *   Else (fallback to request language): "How can I help you?" (or use `$request.language_code` to pick)
    *   **Routes:**
        *   **Route 1: Ask General Question:**
            *   Intent: `AskGeneralQuestion`. Training phrases:
                *   "गेहूं की बुवाई कब करें" (When to sow wheat)
                *   "टमाटर में कौन सी खाद डालें" (Which fertilizer for tomatoes)
                *   "माझ्या पिकाला कीड लागली आहे" (My crop is infested)
                *   "ऊस कसा लावावा" (How to plant sugarcane)
                *   Use `@sys.any` to capture the full query as a parameter (e.g., `user_query`).
            *   Parameter: `user_query` (Type: `@sys.any`)
            *   Fulfillment: Call Webhook (see section 3).
            *   Transition: Stay on `MainInteractionPage` or go to a `DisplayAnswerPage`.
        *   **Route 2: Ask Weather:**
            *   Intent: `AskWeather`. Training phrases:
                *   "आज मौसम कैसा है" (How is the weather today)
                *   "बारिश कब होगी" (When will it rain)
                *   "हवामान कसे आहे" (How is the weather)
                *   "पाऊस कधी येणार" (When will the rain come)
                *   (Optional) Parameter: `location` (Type: `@sys.location`). If not provided, ask for it.
            *   Fulfillment: Call Webhook (see section 4).
            *   Transition: Stay on `MainInteractionPage` or go to a `DisplayWeatherPage`.
        *   **Route 3: Manual Language Change:**
            *   Intent: `ChangeLanguageToHindi`. Training phrases: "हिंदी में बोलो", "भाषा हिंदी करो".
            *   Condition: `$intent.name = "ChangeLanguageToHindi"`
            *   Parameter Preset: `session.params.selected_language` = `hi`
            *   Fulfillment: Agent says: "ठीक है, अब मैं हिंदी में बात करूंगा।"
            *   Transition: `MainInteractionPage`.
            *   (Similarly for `ChangeLanguageToMarathi`)

**D. Set up Phone Gateway Integration:**

1.  Go to "Manage" tab -> "Integrations".
2.  Click "Connect" on "Dialogflow CX Phone Gateway".
3.  Click "+ Create new".
4.  Select Country Code (e.g., India +91).
5.  It will assign you a phone number. Click "Save".
    *   **Note:** This number is for testing and may have limitations. For production, you'd look into porting your own number or more robust telephony options.

---

## 3. Guide: Using Gemini via Webhook

**A. Create a Cloud Function (Python):**

1.  Go to Google Cloud Console -> Cloud Functions.
2.  Click "Create Function".
3.  Configure:
    *   Environment: `2nd gen` (recommended)
    *   Function Name: `agri-assistant-webhook`
    *   Region: Same as your Dialogflow agent (e.g., `us-central1`)
    *   Authentication: "Allow unauthenticated invocations" (Dialogflow will call this. For production, you might secure it and have Dialogflow authenticate, but for simplicity, start with unauthenticated).
    *   Runtime: `Python 3.10` (or newer)
    *   Source Code: Inline editor or ZIP upload.
    *   Entry point: `handle_webhook` (or your function name)

**B. `main.py` for the Cloud Function:**

```python
import functions_framework
import json
from vertexai.preview.generative_models import GenerativeModel, Part
import os
import requests # For OpenWeatherMap

# --- Configuration ---
PROJECT_ID = os.environ.get("GCP_PROJECT") # Automatically set by Cloud Functions
LOCATION = "us-central1" # Or your Gemini model region
GEMINI_MODEL_NAME = "gemini-1.0-pro" # Or "gemini-1.5-pro-preview-0409" if available and you need more context

# OpenWeatherMap API configuration (replace with your key)
OPENWEATHER_API_KEY = "YOUR_OPENWEATHERMAP_API_KEY"
OPENWEATHER_API_URL = "http://api.openweathermap.org/data/2.5/weather"

def get_gemini_response(query: str, language_code: str) -> str:
    """Generates a response from Gemini."""
    try:
        model = GenerativeModel(GEMINI_MODEL_NAME)
        
        # Tailor the prompt for agricultural context and desired tone
        # Instruct Gemini about the language of the answer
        language_instruction = ""
        if language_code.startswith("hi"):
            language_instruction = "Please answer in simple, friendly Hindi."
        elif language_code.startswith("mr"):
            language_instruction = "Please answer in simple, friendly Marathi."
        else: # Default or English
            language_instruction = "Please answer in simple, friendly English."

        prompt = f"""
        You are a helpful and friendly agricultural assistant for Indian farmers.
        {language_instruction}
        Do not use very complex or technical terms unless absolutely necessary, and if so, explain them simply.
        Keep your answers concise and actionable.

        Farmer's query: "{query}"
        
        Your answer:
        """
        
        response = model.generate_content(prompt)
        
        if response.candidates and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text
        else:
            # Handle cases where Gemini might not return a valid text response
            # (e.g., safety-filtered, or unexpected structure)
            print(f"Warning: Gemini response was empty or malformed for query: {query}")
            if language_code.startswith("hi"):
                return "मुझे क्षमा करें, मैं अभी इसका उत्तर नहीं दे पा रहा हूँ।"
            elif language_code.startswith("mr"):
                return "मला माफ करा, मी सध्या याचे उत्तर देऊ शकत नाही."
            else:
                return "I'm sorry, I couldn't generate an answer for that right now."

    except Exception as e:
        print(f"Error calling Gemini: {e}")
        if language_code.startswith("hi"):
            return "तकनीकी समस्या के कारण मैं अभी जवाब नहीं दे पा रहा हूँ।"
        elif language_code.startswith("mr"):
            return "तांत्रिक समस्येमुळे मी सध्या प्रतिसाद देऊ शकत नाही."
        else:
            return "I'm having trouble connecting to my knowledge base at the moment."


def get_weather_info(location_query: str, language_code: str) -> str:
    """Fetches and formats weather information."""
    params = {
        'q': location_query,
        'appid': OPENWEATHER_API_KEY,
        'units': 'metric', # For Celsius
        'lang': language_code.split('-')[0] # 'hi', 'mr', 'en'
    }
    try:
        response = requests.get(OPENWEATHER_API_URL, params=params)
        response.raise_for_status() # Raise an exception for HTTP errors
        data = response.json()

        if data.get("cod") != 200: # OpenWeatherMap specific error code
             error_message = data.get("message", "unknown error")
             print(f"OpenWeatherMap API error for {location_query}: {error_message}")
             if language_code.startswith("hi"):
                return f"{location_query} के लिए मौसम की जानकारी प्राप्त करने में असमर्थ।"
             elif language_code.startswith("mr"):
                return f"{location_query} साठी हवामान माहिती मिळवता आली नाही."
             else:
                return f"Sorry, I couldn't get weather information for {location_query}."


        city_name = data.get("name", location_query)
        description = data['weather'][0]['description']
        temp = data['main']['temp']
        humidity = data['main']['humidity']
        wind_speed = data['wind']['speed'] # m/s

        if language_code.startswith("hi"):
            weather_report = (
                f"{city_name} में अभी मौसम: {description}. "
                f"तापमान {temp}°C, नमी {humidity}%, और हवा की गति {wind_speed} मीटर प्रति सेकंड है।"
            )
        elif language_code.startswith("mr"):
            weather_report = (
                f"{city_name} मधील सध्याचे हवामान: {description}. "
                f"तापमान {temp}°C, आर्द्रता {humidity}%, आणि वाऱ्याचा वेग {wind_speed} मीटर प्रति सेकंद आहे."
            )
        else: # Default English
            weather_report = (
                f"The current weather in {city_name} is: {description}. "
                f"The temperature is {temp}°C, humidity is {humidity}%, and wind speed is {wind_speed} m/s."
            )
        return weather_report

    except requests.exceptions.RequestException as e:
        print(f"Error calling OpenWeatherMap API: {e}")
        if language_code.startswith("hi"):
            return "मौसम सेवा से संपर्क करने में समस्या आ रही है।"
        elif language_code.startswith("mr"):
            return "हवामान सेवेशी संपर्क साधण्यात समस्या येत आहे."
        else:
            return "I'm having trouble connecting to the weather service."
    except Exception as e:
        print(f"Unexpected error in get_weather_info: {e}")
        return "An unexpected error occurred while fetching weather."


@functions_framework.http
def handle_webhook(request):
    """Handles webhook requests from Dialogflow CX."""
    request_json = request.get_json(silent=True)
    tag = request_json.get('fulfillmentInfo', {}).get('tag')
    
    # --- Language Determination ---
    # 1. From session parameter (if explicitly set by user)
    # 2. From Dialogflow's detected language for the query
    # 3. Default to a fallback (e.g., Hindi or English)
    
    session_params = request_json.get('sessionInfo', {}).get('parameters', {})
    selected_language = session_params.get('selected_language')
    
    # If not set by user, use Dialogflow's detected language for the current turn
    if not selected_language:
        selected_language = request_json.get('languageCode', 'hi-IN') # Default to Hindi if not found
    
    # Ensure it's just the base code (e.g., 'hi', 'mr') for some API calls
    base_language_code = selected_language.split('-')[0]


    # --- Logging (Optional, to Firestore or Cloud Logging) ---
    user_query_text = request_json.get('text', '') # Farmer's transcribed text
    print(f"Received query: '{user_query_text}', Detected Lang: {selected_language}, Intent Tag: {tag}")
    # Example: log_to_firestore(user_id, user_query_text, detected_language, intent_tag, response_text)

    fulfillment_response_messages = []

    if tag == 'agri_query_gemini':
        user_query = request_json.get('sessionInfo', {}).get('parameters', {}).get('user_query', '')
        if not user_query and user_query_text: # Fallback to direct text if param not filled
             user_query = user_query_text

        if user_query:
            gemini_answer = get_gemini_response(user_query, selected_language)
            fulfillment_response_messages.append({"text": {"text": [gemini_answer]}})
        else:
            no_query_msg = "आपने क्या पूछा, मैं समझ नहीं पाया।"
            if selected_language.startswith("mr"):
                no_query_msg = "तुम्ही काय विचारले ते मला समजले नाही."
            fulfillment_response_messages.append({"text": {"text": [no_query_msg]}})

    elif tag == 'weather_query_api':
        location_param = request_json.get('sessionInfo', {}).get('parameters', {}).get('location', '')
        
        # If location is a struct (from @sys.location), extract city.
        # This can be complex as @sys.location can return various fields.
        # For simplicity, we assume it might be a string or a dict with 'city'.
        location_name = ""
        if isinstance(location_param, dict):
            location_name = location_param.get('city', location_param.get('locality', location_param.get('administrative_area_level_1', '')))
            if not location_name: # Fallback if city is not found
                 # Try to infer from original text if available
                 location_name = request_json.get('text', '') # Simplistic, might need better parsing
        elif isinstance(location_param, str):
            location_name = location_param

        if not location_name: # If location is still not found, ask for it.
            ask_location_msg = "कृपया बताएं आप किस जगह का मौसम जानना चाहते हैं?"
            if selected_language.startswith("mr"):
                ask_location_msg = "तुम्हाला कोणत्या ठिकाणचे हवामान जाणून घ्यायचे आहे, कृपया सांगा?"
            
            fulfillment_response_messages.append({
                "text": {"text": [ask_location_msg]},
                # This signals Dialogflow to reprompt for the 'location' parameter.
                # Make sure your 'location' parameter in Dialogflow is set to required.
                "parameter_name": "location", # This might not be standard, check DF CX docs for re-prompting parameters
                                             # A better way is to transition to a page that specifically asks for location.
            })
            # Or, more robustly, design your Dialogflow to transition to a page that specifically asks for location
            # if the location parameter is not filled. For now, we'll just return a message.

        else:
            weather_report = get_weather_info(location_name, selected_language)
            fulfillment_response_messages.append({"text": {"text": [weather_report]}})
            
    else:
        unknown_tag_msg = "मैं इस अनुरोध को समझ नहीं सका।"
        if selected_language.startswith("mr"):
            unknown_tag_msg = "मला ही विनंती समजली नाही."
        fulfillment_response_messages.append({"text": {"text": [unknown_tag_msg]}})


    # Construct the response for Dialogflow CX
    response_json = {
        "fulfillment_response": {
            "messages": fulfillment_response_messages
        },
        "session_info": { # Pass back session parameters if they were modified
            "parameters": {
                "selected_language": selected_language
            }
        }
    }
    
    return json.dumps(response_json)

# Placeholder for Firestore logging
# def log_to_firestore(user_id, query, lang, intent, response):
#   from google.cloud import firestore
#   db = firestore.Client()
#   doc_ref = db.collection(u'agri_assistant_logs').document()
#   doc_ref.set({
#       u'timestamp': firestore.SERVER_TIMESTAMP,
#       u'user_id': user_id, # You might get this from caller ID or generate a session ID
#       u'query': query,
#       u'language': lang,
#       u'intent_tag': intent,
#       u'response': response
#   })
#   print("Logged to Firestore")

```

**C. `requirements.txt` for the Cloud Function:**

```txt
functions-framework>=3.0.0
google-cloud-aiplatform>=1.25.0  # For Vertex AI / Gemini
requests>=2.25.0
# google-cloud-firestore # If you add Firestore logging
```

**D. Deploy the Cloud Function:**

1.  Back in the Cloud Function creation UI, select "ZIP upload" or "Inline editor" and provide the code.
2.  Make sure the entry point matches your function name (`handle_webhook`).
3.  **Set Environment Variables (under Runtime, build and connections settings -> Runtime environment variables):**
    *   `GCP_PROJECT`: Your Project ID (though often auto-set)
    *   `OPENWEATHER_API_KEY`: Your_Actual_OpenWeatherMap_API_Key
4.  Ensure the **Service Account** for the Cloud Function (usually `YourProjectID@appspot.gserviceaccount.com` for 1st gen, or a custom one for 2nd gen) has the "Vertex AI User" role (to access Gemini) and "Cloud Functions Invoker" (if you decide to secure it later).
5.  Click "Deploy". Once deployed, copy the **Trigger URL**.

**E. Connect Webhook in Dialogflow CX:**

1.  In Dialogflow CX, go to "Manage" tab -> "Webhooks".
2.  Click "+ Create".
3.  Name: `GeminiAgriWebhook`
4.  Webhook URL: Paste the Trigger URL of your Cloud Function.
5.  (Optional) Configure Authentication if you secured your function.
6.  Save.

**F. Use Webhook in your Flow (`MainInteractionPage`):**

1.  Go to the `MainInteractionPage` (or the relevant page for general queries).
2.  Select the Route for `AskGeneralQuestion`.
3.  In the "Fulfillment" section:
    *   Enable Webhook: Select `GeminiAgriWebhook`.
    *   **Tag:** Enter `agri_query_gemini`. This tag is used in the Cloud Function to know what kind of request it is.
4.  When the webhook responds, the text will be spoken back to the user.

---

## 4. Connect Weather Alerts

This is already partially handled in the `handle_webhook` function above if the tag is `weather_query_api`.

**A. Dialogflow CX Setup for Weather:**

1.  On `MainInteractionPage`, select the Route for `AskWeather`.
2.  **Parameter:**
    *   Ensure you have a parameter `location` (Type: `@sys.location`).
    *   Mark it as "Required" if you want Dialogflow to automatically ask for it if the user doesn't provide it.
    *   Prompt for `location` if missing:
        *   Hindi: "आप किस जगह का मौसम जानना चाहते हैं?"
        *   Marathi: "तुम्हाला कोणत्या जागेचे हवामान जाणून घ्यायचे आहे?"
3.  In the "Fulfillment" section for this route:
    *   Enable Webhook: Select `GeminiAgriWebhook` (your same function).
    *   **Tag:** Enter `weather_query_api`.

**B. Cloud Function Logic (already in `main.py` above):**

*   The `get_weather_info` function fetches data from OpenWeatherMap.
*   It formats the response in the detected language.
*   Ensure you have `OPENWEATHER_API_KEY` set in your Cloud Function's environment variables. Get a free API key from [OpenWeatherMap](https://openweathermap.org/appid).

**Alternative: Google Weather API**
Google does not have a direct, publicly available "Google Weather API" in the same way OpenWeatherMap does. Weather information is often integrated into Google Search, Google Assistant, or through Google Maps APIs (like the Places API which *can* sometimes include weather details, but it's not its primary purpose). For a dedicated weather API, OpenWeatherMap is a common choice. If Google were to release a more direct weather API via Vertex AI or another service, you could adapt the function.

---

## 5. Deploy and Test the Full Pipeline

1.  **Deploy Cloud Function:** If not already done, deploy `agri-assistant-webhook`.
2.  **Configure Dialogflow CX:**
    *   Ensure webhooks are set up for the correct routes with the correct tags.
    *   Ensure language settings (auto-detection, TTS voices) are correct.
    *   Ensure intents (`AskGeneralQuestion`, `AskWeather`, `select_hindi`, `select_marathi`, `ChangeLanguageTo...`) have sufficient training phrases in both Hindi and Marathi.
3.  **Test with Dialogflow CX Simulator:**
    *   Click "Test Agent" in the Dialogflow CX console.
    *   Type or speak queries in Hindi and Marathi.
    *   Verify:
        *   Language detection (check `session.params.selected_language` or `$request.language_code`).
        *   Intent matching.
        *   Parameter extraction (e.g., `user_query`, `location`).
        *   Webhook calls (check Cloud Function logs for incoming requests and any errors).
        *   Responses from Gemini and Weather API.
        *   TTS output quality.
4.  **Test with Phone Call:**
    *   Call the phone number provided by the Dialogflow CX Phone Gateway.
    *   Speak naturally.
    *   Test all features:
        *   Initial language selection.
        *   General agricultural queries.
        *   Weather queries (with and without specifying location initially).
        *   Manual language change mid-conversation.
5.  **Debugging:**
    *   **Dialogflow CX:** Use the simulator's "Interaction logs" and "Flow execution details".
    *   **Cloud Functions:** Check logs in Google Cloud Logging. Add `print()` statements in your Python code for debugging.
    *   **Gemini:** Check for API errors or safety filter responses.
    *   **OpenWeatherMap:** Check if your API key is valid and not rate-limited.

---

## 6. Ensure Support for Hindi and Marathi

*   **Dialogflow Agent Languages:** Already configured in step 2A.
*   **STT Auto Language Detection:** Configured in step 2B. This is crucial. It will try to detect from the configured list (`hi-IN`, `mr-IN`).
*   **TTS Voices:** Specific WaveNet voices for Hindi and Marathi selected in step 2B.
*   **Intent Training Phrases:** Provide ample training phrases in *both* Hindi and Marathi for *all* relevant intents.
    *   Example for `AskGeneralQuestion`:
        *   Hindi: "गेंहू के लिए सबसे अच्छा बीज कौन सा है?", "धान की फसल में पानी कब देना चाहिए?"
        *   Marathi: "गव्हासाठी सर्वोत्तम बियाणे कोणते?", "भात पिकाला पाणी कधी द्यावे?"
*   **Gemini Prompting for Language:** The Cloud Function's `get_gemini_response` includes instructions like: `"Please answer in simple, friendly Hindi."` or `"Please answer in simple, friendly Marathi."` based on `selected_language`.
*   **Weather API Language Parameter:** The `get_weather_info` function passes `lang='hi'` or `lang='mr'` to the OpenWeatherMap API.
*   **Static Text in Dialogflow:** For any agent responses directly configured in Dialogflow (not from webhook), ensure you provide variants for each language.
    *   Example: If you have a welcome message:
        *   Click the text bubble.
        *   Click "+ Add language variant".
        *   Select Hindi, type the Hindi text.
        *   Select Marathi, type the Marathi text.
*   **Manual Language Override:**
    *   Create intents like `SwitchToHindi` ("हिंदी में बात करो", "भाषा बदलो हिंदी") and `SwitchToMarathi` ("मराठीत बोल", "भाषा बदल मराठी").
    *   In their fulfillment, set the `session.params.selected_language` parameter to `hi` or `mr` respectively.
    *   The webhook will then use this `selected_language` for Gemini and Weather API calls.
    *   Agent response: "ठीक है, अबसे मैं हिंदी में बात करूँगा।" / "ठीक आहे, आतापासून मी मराठीत बोलेन."

---

## Optional: Logging Interactions

The `log_to_firestore` placeholder function in `main.py` can be implemented:

1.  **Enable Firestore API** in your GCP project.
2.  Create a Firestore database in Native mode or Datastore mode.
3.  Add `google-cloud-firestore` to `requirements.txt`.
4.  Uncomment and complete the `log_to_firestore` function.
5.  Call `log_to_firestore` from your `handle_webhook` function, passing relevant data.
    *   You'll need a `user_id`. For phone calls, this could be the (hashed) caller ID if available and compliant with privacy, or a generated session ID from Dialogflow. `request_json.get('sessionInfo', {}).get('session')` gives you the Dialogflow session ID.

```python
# In main.py, ensure you have this at the top if using Firestore
# from google.cloud import firestore
# db = None # Initialize globally

# def init_firestore():
# global db
# if db is None:
# db = firestore.Client()

# In handle_webhook, before returning the response:
# init_firestore() # Ensure db client is initialized
# session_id = request_json.get('sessionInfo', {}).get('session', 'unknown_session')
# response_text_to_log = ""
# if fulfillment_response_messages and fulfillment_response_messages[0].get('text', {}).get('text'):
#     response_text_to_log = fulfillment_response_messages[0]['text']['text'][0]

# log_to_firestore(
#     user_id=session_id, # Using Dialogflow session ID
#     query=user_query_text,
#     lang=selected_language,
#     intent_tag=tag,
#     response=response_text_to_log
# )
```

---

This is a comprehensive plan. Start by setting up the Dialogflow agent and its basic flows, then implement the Cloud Function and test one feature (e.g., general query with Gemini). Gradually add weather, language switching, and logging. Good luck!
