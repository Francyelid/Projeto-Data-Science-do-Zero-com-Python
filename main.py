# Resumo do Python

# Primeiro programa em python
nome = input("Qual é o seu nome? ")
print(f"Hello {nome}")

# Desafio Python
# Tendo como dados de entrada a altura de uma pessoa, 
# construa um algoritmo que calcule seu peso ideal, usando a seguinte fórmula: (72.7*altura) - 58
# altura = float(input("Qual é a sua altura?"))
# PesoIdeal = (72.7*altura) - 58
# print(f"O seu peso ideal deveria ser: {PesoIdeal}")

# DESAFIO CONDICIONAIS
# Faça um Programa que pergunte em que turno você estuda. Peça para digitar M-matutino ou V-Vespertino 
# ou N- Noturno. Imprima a mensagem "Bom Dia!", "Boa Tarde!" ou "Boa Noite!" ou "Valor Inválido!", conforme o caso.
turno = input("Qual turno você estuda?")

if( turno == "M"):
    print("Bom dia!")

elif( turno == "V"):
    print("Boa tarde!")

elif( turno == "N"):
    print("Boa noite!")

else:
    print("Valor inválido")