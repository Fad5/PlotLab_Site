from django.shortcuts import render

def vibra_protocol(request):
    return render(request, 'protocol/vibra_protocol.html')

def press_protocol(request):
    return render(request, 'protocol/press_protocol.html')

def protocol(request):
    return render(request, 'protocol/protocol.html')