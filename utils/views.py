from django.shortcuts import render

def weather_view(request):
    return render(request, 'weather.html')  # Ensure this matches your template
