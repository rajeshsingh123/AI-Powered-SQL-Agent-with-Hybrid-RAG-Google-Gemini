
# chatbot/views.py
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from . import bot
import json
import uuid
import logging
import time
import threading

logger = logging.getLogger('chatbot')

def chat_ui(request):
    return render(request, "chat.html")

@csrf_exempt
@require_POST
def chat(request):
    try:
        start_time = time.time()
        session_id = request.headers.get('X-Session-ID', str(uuid.uuid4()))
        thread_id = threading.current_thread().ident  # Log thread ID
        logger.info(f"Request started for session_id: {session_id}, thread: {thread_id}")

        if session_id not in request.session._session:
            request.session[session_id] = {'chat_history': []}
        chat_history = request.session[session_id].get('chat_history', [])

        data = json.loads(request.body.decode('utf-8'))
        user_query = data.get('user_query', '')
        logger.info(f"User query received: '{user_query}' for session_id: {session_id}, thread: {thread_id}")


        if not user_query:
            logger.info(f"Returning history only for session_id: {session_id}, thread: {thread_id}")
            return JsonResponse({
                'response': None,
                'chat_history': chat_history
            })

        # Simulate some processing time to exaggerate overlap
        time.sleep(2)  # Add artificial delay for testing
        response = bot.process_user_query(user_query, chat_history)
        request.session[session_id]['chat_history'] = chat_history
        request.session.modified = True

        duration = time.time() - start_time
        logger.info(f"Request completed for session_id: {session_id}, thread: {thread_id}, duration: {duration:.3f}s")

        return JsonResponse({
            'response': response,
            'chat_history': chat_history
        })

    except Exception as ex:
        logger.error(f"Error in request for session_id: {session_id}: {str(ex)}")
        return JsonResponse({'error': str(ex)}, status=500)
    

    
# from django.shortcuts import render
# # Create your views here.
# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# from django.views.decorators.http import require_POST
# from . import bot
# import uuid
# from django.shortcuts import render

# def chat_ui(request):
#     return render(request, "chat.html")

# import json
# import uuid
# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# from django.views.decorators.http import require_POST

# @csrf_exempt
# @require_POST
# def chat(request):
#     try:
#         # Extract session ID from request headers or create a new one
#         session_id = request.headers.get('X-Session-ID', str(uuid.uuid4()))
#         request.session['session_id'] = session_id

#         # Parse JSON request body
#         data = json.loads(request.body.decode('utf-8'))
#         user_query = data.get('user_query', '')

#         # Initialize chat history if not present
#         if 'chat_history' not in request.session:
#             request.session['chat_history'] = []

#         chat_history = request.session['chat_history']

#         # Debugging: Print user query
#         print("User query received:", user_query)
#         if not user_query:
#             return JsonResponse({
#                 'response': None,
#                 'chat_history': chat_history
#             })

#         # Process user query using chatbot logic
#         response = bot.process_user_query(user_query, chat_history)

#         # Update session history and force session save
#         # chat_history.append({'user': user_query, 'bot': response})
#         # request.session['chat_history'] = chat_history
#         request.session['chat_history'] = chat_history
#         # request.session.modified = True
#         request.session.modified = True  # Ensure changes are saved
#         return JsonResponse({
#             'response': response,
#             'chat_history': chat_history
#         })

#         # return JsonResponse({'response': response})

#     except Exception as ex:
#         print("Error:", ex)
#         return JsonResponse({'error': str(ex)}, status=500)


