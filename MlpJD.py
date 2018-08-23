import math
import numpy as np
np.set_printoptions(linewidth=127)


x = np.array([[1, 1, -1, 1],
              [1, 1, -1, -1],
              [1, -1, -1, -1],
              [-1, -1, 1, 1],])

yd = np.array([[1, -1, -1, -1],
               [-1, 1, -1, -1],
               [-1, -1, 1, -1],
               [-1, -1, -1, 1]])

w1 = np.array([1, -1, -1, 1])
wn = np.array([[1, 1],
              [-1, 1],
              [1, -1],
              [-1, -1]])

w = np.array([[1, 1],
              [-1, 1],
              [1, -1],
              [-1, -1]])

em_guardado = np.array([])

instante = 0
apoio = 0
epoca = 1
erroMaxima = 0.27
#erroMaxima = 0.05


while instante < len(x):

    #1) Calcular v4 e Y4 separadamente do demais, devido a diferença de tamanho dos vetores.

    v1 = np.sum([x[instante] * w1])
    y1 = (1/(1+math.exp(-0.5*v1)))
    vaux = np.array([1, y1])

    # Calcular V2 a V5

    v = np.array([])
    i = 0
    for apoio in wn:
        vaux2 = np.sum([vaux * wn[i]])
        v = np.append(v, [vaux2])
        i = i+1

    #calcular Y
    y = np.array([])
    i = 0
    for apoio in v:
        yaux = (1 / (1 + math.exp(-0.5 * v[i])))
        y = np.append(y, [yaux])
        i = i+1

    # Calcular o Erro
    e = np.array([])
    i = 0
    for apoio in yd:
        eaux = yd[instante][i] - y[i]
        e = np.append(e, [eaux])
        i = i+1
    em = ((np.sum(e))/len(e))
    em_guardado = np.append(em_guardado, [em*em])

    # Calcular gradientes local do erro
    s = np.array([])
    i=0
    for apoio in e:
        saux = (e[i]*0.5*y[i]*(1-y[i]))
        s = np.append(s, [saux])
        i = i+1

    #calcular Delta W e Wn
    deltaw = np.array([[]])
    i = 0
    for apoio in s:
        deltaAux = 0.5*s[i]*vaux
        deltaw = np.append(deltaw, [deltaAux])
        i = i + 1
    deltaw = (np.split(deltaw, len(s)))
    wn = (np.add(wn, deltaw))


    #passo 5
    s1 = (0.5*y1*(1-y1))
    i = 0
    for apoio in w:
        if i == 0:
            somatorio = np.array([])
        somatorio = np.append(somatorio, [(s[i] * w[i][1])])
        soma = np.sum(somatorio)
        i = i+1
    s1 = s1*soma


    # Passo 6 Atualizar Sinapse do W1
    deltaw1 = np.array([[]])
    deltaw1 = 0.5*s1*x[instante]
    w1 = (np.add(w1, deltaw1))

    print("Rodada =", instante + 1)
    print("W Atualizado:\n", w1)
    print(wn)
    print("Epoca =", epoca)
    instante = instante + 1

    if instante == len(x):
        emq = np.sum(em_guardado)/len(x)
        print("Erro médio Quadratico =", emq)
        print("-------------")
        print("\n")
        epoca = epoca +1
        em_guardado = np.array([])
        if emq > erroMaxima:
            instante = 0





