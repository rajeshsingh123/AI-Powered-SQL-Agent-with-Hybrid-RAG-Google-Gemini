from django.contrib import admin
from django.urls import path
from django.views.decorators.csrf import csrf_exempt
from chatbot.views import chat ,chat_ui # Import the chat view
# from .views import chat, chat_ui

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/chat/', csrf_exempt(chat)),  # Add the chat endpoint
    path("chat-ui/", chat_ui, name="chat-ui"),
]