# Desafio Python
# Tendo como dados de entrada a altura de uma pessoa, 
# construa um algoritmo que calcule seu peso ideal, usando a seguinte fórmula: (72.7*altura) - 58
altura = float(input("Qual é a sua altura?"))
PesoIdeal = (72.7*altura) - 58
print(f"O seu peso ideal deveria ser: {PesoIdeal}")

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

# DESAFIO REPETIÇÃO
# A série de Fibonacci é formada pela seqüência 1,1,2,3,5,8,13,21,34,55,... Faça um programa capaz de gerar a série até o n−ésimo termo.
# qual numero voce deseja receber a sequencia fibonacci? 34
sequencia = int(input("Qual numero voce deseja receber a sequencia fibonacci?"))

proximo = 0
anterior = 0

while(proximo <= sequencia):
    print(proximo)
    proximo = anterior + proximo
    anterior = proximo - anterior

    if( proximo == 0):
        proximo = 1

# DESAFIO FUNÇÕES
# Reverso do número. Faça uma função que retorne o reverso de um número inteiro informado. Por exemplo: 127 -> 721.

def inverte(x):
    return x[::-1]

num = input("Informe um número: ")

print(inverte(num))
