"""
Created on Tue Jun 30 12:33:44 2022

@author: TCHAPPI OSEE
"""
# Résolution du problème du sac à dos ou KP (Knapsack Problem) à l'aide d'algorithme génétique
# import de la librairie

from pydoc import doc
import numpy as np #utilisation des calculs matriciels
import pandas as pd #générer les fichiers csv
import random as rd #génération de nombre al�atoire
from random import randint # génération des nombres al�atoires  
import matplotlib.pyplot as plt
import time
import sys


# Donn�es du probl�me (g�n�r�es al�atoirement)
nombre_objets = 30
capacite_max = 30    #La capacité du sac 

#param�tres de l'algorithme g�n�tique
nbr_generations = 100 # nombre de g�n�rations


#ID_objets = np.arange(0, nombre_objets) #ID des objets � mettre dans le sac de 1 � 10
#poids = np.random.randint(1, 15, size=nombre_objets) # Poids des objets g�n�r�s al�atoirement
#valeur = np.random.randint(50, 350, size=nombre_objets) # Valeurs des objets g�n�r�es al�atoirement

Instance = pd.read_csv("Q2-Instance4.csv")
ID_objets = np.array(Instance.loc[:,"Objets"])
ID_objets = ID_objets.astype(int)
poids = np.array(Instance.loc[:,"Poids"])
poids = poids.astype(int)
valeur = np.array(Instance.loc[:,"Valeurs"])
valeur = valeur.astype(int)

print('La liste des objets est la suivante :')
print('ID_objet   Poids   Valeur')
for i in range(ID_objets.shape[0]):
    print(f'{ID_objets[i]} \t {poids[i]} \t {valeur[i]}')
print()


#calcule le poids d'une solution
def cal_poids(poids, population):
    fitness = np.empty(population.shape[0])
    for i in range(population.shape[0]):
        fitness[i] = np.sum(population[i] * poids)

    return fitness.astype(int) 

#corriger un individus
def corriger_individu(individu, capacite, poids):
    if(np.sum(individu * poids) <= capacite):
        return individu
    taille = len(individu)
    ind = np.zeros(taille)  # Créer un individu initialisé avec des zéros
    indices = np.random.permutation(taille)
    indice,poids_total = 0,0

    while(indice < taille and (poids_total + individu[indices[indice]]*poids[indices[indice]]) <= capacite):
        ind[indices[indice]] = individu[indices[indice]]
        poids_total += individu[indices[indice]]*poids[indices[indice]]
        indice += 1
    return ind

# Cr�er la population initiale
solutions_par_pop = 8 #10 #la taille de la population 
pop_size = (solutions_par_pop, ID_objets.shape[0])
population_initiale = np.random.randint(2, size = pop_size)
for i in range(population_initiale.shape[0]) :
    population_initiale[i,:] = corriger_individu(population_initiale[i], capacite_max, poids)

population_initiale = population_initiale.astype(int)
print(f'Taille de la population: {pop_size}')
print(f'Population Initiale: \n{population_initiale}')
print(f'Poids population initiale : {cal_poids(poids, population_initiale)}') 


def cal_fitness(poids, valeur, population, capacite):
    fitness = np.empty(population.shape[0])

    for i in range(population.shape[0]):
        S1 = np.sum(population[i] * valeur)
        S2 = np.sum(population[i] * poids)

        if S2 <= capacite:
            fitness[i] = S1
        else:
            fitness[i] = 0

    return fitness.astype(int)  

def selection(fitness, nbr_parents, population):
    fitness = list(fitness)
    parents = np.empty((nbr_parents, population.shape[1]))

    for i in range(nbr_parents):
        indice_max_fitness = np.where(fitness == np.max(fitness))
        parents[i,:] = population[indice_max_fitness[0][0], :]
        fitness[indice_max_fitness[0][0]] = -999999

    return parents

def croisement(parents, nbr_enfants, capacite):
    enfants = np.empty((nbr_enfants, parents.shape[1]))
    point_de_croisement = int(parents.shape[1]/2) #croisement au milieu
    taux_de_croisement = 0.8
    i = 0

    while (i < nbr_enfants): #parents.shape[0]
        indice_parent1 = i%parents.shape[0]
        indice_parent2 = (i+1)%parents.shape[0]
        x = rd.random()
        if x > taux_de_croisement: # parents st�riles
            continue
        indice_parent1 = i%parents.shape[0]
        indice_parent2 = (i+1)%parents.shape[0]
        # On va assurer que les enfants soient des solution réalisable
        enfant = np.empty(parents.shape[1])
        enfant[0:point_de_croisement] = parents[indice_parent1,0:point_de_croisement]
        enfant[point_de_croisement:] = parents[indice_parent2,point_de_croisement:]
        enfants[i,:] = corriger_individu(enfant, capacite_max, poids)
        i += 1
    return enfants

# La mutation consiste � inverser le bit
def mutation(enfants):
    mutants = np.empty((enfants.shape))
    mutant = np.empty(enfants.shape[1])
    taux_mutation = 0.5
    for i in range(mutants.shape[0]):
        
        random_valeur = rd.random()
        #mutants[i,:] = enfants[i,:]
        mutant = enfants[i]
        if random_valeur > taux_mutation:
            continue
        int_random_valeur = randint(0,enfants.shape[1]-1) #choisir al�atoirement le bit � inverser    
        if mutant[int_random_valeur] == 0:
            mutant[int_random_valeur] = 1
        else:
            mutant[int_random_valeur] = 0
        mutant = corriger_individu(mutant, capacite_max, poids)
        mutants[i,:] = mutant
    return mutants  

def optimize(poids, valeur, population, pop_size, nbr_generations, capacite):
    sol_opt, historique_fitness = [], []
    nbr_parents = pop_size[0]//2
    nbr_enfants = pop_size[0] - nbr_parents 
    for _ in range(nbr_generations):
        fitness = cal_fitness(poids, valeur, population, capacite)
        historique_fitness.append(fitness)
        parents = selection(fitness, nbr_parents, population)
        enfants = croisement(parents, nbr_enfants, capacite_max)
        mutants = mutation(enfants)
        population[0:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = mutants

    print(f'Voici la dernière génération de la population: \n{population}\n') 
    fitness_derniere_generation = cal_fitness(poids, valeur, population, capacite)      
    print(f'Fitness de la dernière génération: \n{fitness_derniere_generation}\n')
    max_fitness = np.where(fitness_derniere_generation == np.max(fitness_derniere_generation))
    sol_opt.append(population[max_fitness[0][0],:])

    return sol_opt, historique_fitness


#lancement de l'algorithme g�n�tique
start_time = time.time()
sol_opt, historique_fitness = optimize(poids, valeur, population_initiale, pop_size, nbr_generations, capacite_max)

end_time = time.time()
execution_time = end_time - start_time

print("Temps d'exécution :", execution_time, "secondes")
#affichage du r�sultat
print('La solution optimale est:')
# print(sol_opt)
print('objets n°', [i for i, j in enumerate(sol_opt[0]) if j!=0])

# print(np.asarray(historique_fitness).shape)
print(f'Avec une valeur de {np.amax(historique_fitness)} et un poids de {np.sum(sol_opt * poids)} kg')
print('Les objets qui maximisent la valeur contenue dans le sac sans le déchirer :')
objets_selectionnes = ID_objets * sol_opt
for i in range(objets_selectionnes.shape[1]):
    if objets_selectionnes[0][i] != 0:
        print(f'{objets_selectionnes[0][i]}')

     
historique_fitness_moyenne = [np.mean(fitness) for fitness in historique_fitness]
historique_fitness_max = [np.max(fitness) for fitness in historique_fitness]
plt.plot(list(range(nbr_generations)), historique_fitness_moyenne, label='Valeurs moyennes')
plt.plot(list(range(nbr_generations)), historique_fitness_max, label='Valeur maximale')
plt.legend()
plt.title('Evolution de la Fitness à travers les générations')
plt.xlabel('Générations')
plt.ylabel('Fitness')
plt.show()