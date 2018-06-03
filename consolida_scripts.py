import os

scripts = os.listdir("./scripts")

script_final = []

with open('view.py', 'r') as dados:
    script_final.append(dados.read())

for arquivo in scripts:
    with open('./scripts/' + arquivo, 'r') as dados:
        script_final.append(dados.read())

with open('Main.py', 'r') as dados:
    script_final.append(dados.read())

with open("script.py", 'w') as arquivo_final:
    arquivo_final.write("".join(script_final))
