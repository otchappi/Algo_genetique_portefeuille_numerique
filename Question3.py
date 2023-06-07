"""
Created on Tue Jun 06 08:33:44 2023

@author: TCHAPPI OSEE
"""
# Application du problème du sac à dos ou KP (Knapsack Problem) à l'aide d'algorithme génétique
#Sut la gestion du portefeuille
# import de la librairie

from pydoc import doc
import numpy as np #utilisation des calculs matriciels
import pandas as pd #générer les fichiers csv
import random as rd #génération de nombre aléatoire
from random import randint # génération des nombres aléatoires  
import matplotlib.pyplot as plt
import time
import sys


# Données du probléme (générées aléatoirement)
nombre_titre = 30
budget = 3000    #Le budget

#paramètres de l'algorithme génétique
nbr_generations = 100 # nombre de générations


#ID_titres = np.arange(0, nombre_titre) #ID des titres à acheter
#prix_achat = np.random.randint(1, 1000, size=nombre_titre) # Prix d'achat des titres générés aléatoirement
#valeur = np.random.randint(150, 500, size=nombre_titre) # Valeurs des titres générées aléatoirement

#mid_term_marks = {"Titres": ID_titres,
#                  "Prix_achat": prix_achat,
#                  "Valeurs": valeur}
#mid_term_marks_df = pd.DataFrame(mid_term_marks)
#mid_term_marks_df.to_csv("Q3-Instance4.csv", index=False)

Instance = pd.read_csv("Q3-Instance3.csv")
ID_titres = np.array(Instance.loc[:,"Titres"])
ID_titres = ID_titres.astype(int)
prix_achat = np.array(Instance.loc[:,"Prix_achat"])
prix_achat = prix_achat.astype(int)
valeur = np.array(Instance.loc[:,"Valeurs"])
valeur = valeur.astype(int)


print('La liste des titres est la suivante :')
print('ID_titre   Prix_achat   Valeur')
for i in range(ID_titres.shape[0]):
    print(f'{ID_titres[i]} \t\t {prix_achat[i]} \t {valeur[i]}')
print()

#calcule le prix d'achat total d'une solution
def cal_prix_achat(prix, population):
    pa = np.empty(population.shape[0])
    for i in range(population.shape[0]):
        pa[i] = np.sum(population[i] * prix)

    return pa.astype(int) 

#renvoie l'indice du titre ayant plusieurs actions
def ind_multi_action(individu):
    indices = np.where(individu>1)[0]
    if(len(indices) > 1) :
        if(individu[indices[0]]*valeur[indices[0]] > individu[indices[1]]*valeur[indices[0]]) :
            individu[indices[1]] = rd.randint(0,1)
            return indices[0]
        else:
            individu[indices[0]] = rd.randint(0,1)
            return indices[1]
    else:
        if(len(indices) == 1):
            return indices[0]
    
    return -1
    
#vérifier le facteur risque d'un individu
def verif_risque(prix, individu, budget):
    i = ind_multi_action(individu)
    if(individu[i]*prix[i] > 0.2*budget):
        return False

    return True 

#corriger un individus
def corriger_individu(individu, budget, prix):
    if(np.sum(individu * prix) <= budget and verif_risque(prix, individu, budget)):
        return individu
    taille = len(individu)
    ind = np.zeros(taille)  # Créer un individu initialisé avec des zéros
    indices = np.random.permutation(taille)
    indice,prix_total = 0,0

    if(ind_multi_action(individu) == -1):
        propab_multi_action = 0.75
        x = rd.random()
        if x <= propab_multi_action: # Achat de plusieurs actions d'un titre
            titre = rd.randint(0, nombre_titre-1)
            individu[titre] = rd.randint(1, (0.2*budget)//prix[titre])

    while(indice < taille and (prix_total + individu[indices[indice]]*prix[indices[indice]]) <= budget):
        val = individu[indices[indice]]
        if(val > 1):
            while(val*prix[indices[indice]] > 0.2*budget):
                val -= 1
        ind[indices[indice]] = val
        prix_total += val*prix[indices[indice]]
        indice += 1
    return ind

# Créer la population initiale
solutions_par_pop = 8  #la taille de la population 
pop_size = (solutions_par_pop, ID_titres.shape[0])
population_initiale = np.random.randint(2, size = pop_size)
for i in range(population_initiale.shape[0]) :
    population_initiale[i,:] = corriger_individu(population_initiale[i], budget, prix_achat)

population_initiale = population_initiale.astype(int)
print(f'Taille de la population: {pop_size}')
print(f'Population Initiale: \n{population_initiale}')
print(f'Poids population initiale : {cal_prix_achat(prix_achat, population_initiale)}')


def cal_fitness(prix, valeur, population, budget):
    fitness = np.empty(population.shape[0])

    for i in range(population.shape[0]):
        S1 = np.sum(population[i] * valeur)
        S2 = np.sum(population[i] * prix)

        if S2 <= budget:
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

def croisement(parents, nbr_enfants, budget, prix):
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
        enfants[i,:] = corriger_individu(enfant, budget, prix)
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
        mutant = corriger_individu(mutant, budget, prix_achat)
        mutants[i,:] = mutant
    return mutants  

def optimize(prix, valeur, population, pop_size, nbr_generations, budget):
    sol_opt, historique_fitness = [], []
    nbr_parents = pop_size[0]//2
    nbr_enfants = pop_size[0] - nbr_parents 
    for _ in range(nbr_generations):
        fitness = cal_fitness(prix, valeur, population, budget)
        historique_fitness.append(fitness)
        parents = selection(fitness, nbr_parents, population)
        enfants = croisement(parents, nbr_enfants, budget, prix)
        mutants = mutation(enfants)
        population[0:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = mutants

    print(f'Voici la dernière génération de la population: \n{population}\n') 
    fitness_derniere_generation = cal_fitness(prix, valeur, population, budget)      
    print(f'Fitness de la dernière génération: \n{fitness_derniere_generation}\n')
    max_fitness = np.where(fitness_derniere_generation == np.max(fitness_derniere_generation))
    sol_opt.append(population[max_fitness[0][0],:])

    return sol_opt, historique_fitness


#lancement de l'algorithme g�n�tique
start_time = time.time()
sol_opt, historique_fitness = optimize(prix_achat, valeur, population_initiale, pop_size, nbr_generations, budget)
end_time = time.time()
execution_time = end_time - start_time

print("Temps d'exécution :", execution_time, "secondes")

#affichage du r�sultat
print('La solution optimale est:')
# print(sol_opt)
print('Titres n°', [i for i, j in enumerate(sol_opt[0]) if j!=0])

# print(np.asarray(historique_fitness).shape)
print(f'Avec une valeur de {np.amax(historique_fitness)} et un prix total d\'achat de {np.sum(sol_opt * prix_achat)} €')
print('Nb action par titre qui maximisent la valeur contenue dans le portefeuille sans dépasser le budget :')
print(sol_opt[0])
     
historique_fitness_moyenne = [np.mean(fitness) for fitness in historique_fitness]
historique_fitness_max = [np.max(fitness) for fitness in historique_fitness]
plt.plot(list(range(nbr_generations)), historique_fitness_moyenne, label='Valeurs moyennes')
plt.plot(list(range(nbr_generations)), historique_fitness_max, label='Valeur maximale')
plt.legend()
plt.title('Evolution de la Fitness à travers les générations')
plt.xlabel('Générations')
plt.ylabel('Fitness')
plt.show()