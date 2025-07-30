from django.shortcuts import render

def help(request):
    return render(request, 'help/directory.html')

def vibra_table(request):
    return render(request, 'help/vibra_table.html')

def servo(request):
    return render(request, 'help/servo.html')

def pulsator(request):
    return render(request, 'help/pulsator.html')

def PHP(request):
    return render(request, 'help/Gost_70261—2022.html')

def Gost_59940_2021(request):
    return render(request, 'help/Gost_59940-2021.html')