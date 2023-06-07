# Algo_genetique_portefeuille_numerique
TP1 Algorithme génétique appliqué au Knapsack Problem

Consignes
    A l’issu du TP, l’étudiant produit un compte rendu en pdf accompagné des fichiers .py correspondant
aux questions 2, 3 et 4. L’ensemble est compressé et envoyé dans un fichier NOM_PRENOM.zip sur
Moodle. Mettez votre nom et prénom dans le pdf et aussi en commentaire des programmes. Il y aura
une présentation mi-parcours des résultats obtenus.

Objectif
    Ce TP vise à résoudre le problème du sac à dos connu sous le nom KP (Knapsack Problem) à l’aide des
algorithmes génétiques. Voici une description simple du problème :
Un voleur entre dans une maison en portant un sac à dos qui peut porter un certain poids (30 kg). La
maison dispose de 10 objets de valeurs différentes, chacun ayant un poids. Le dilemme du voleur est
alors de faire une sélection d'objets qui maximise la valeur totale des objets volés sans dépasser le
poids du sac à dos.
    Il existe plusieurs applications du problème du sac à dos. Il est particulièrement utilisé dans la gestion
du portefeuille. Par exemple, avec un budget limité (capacité du sac), le problème consiste à acheter
les titres qui maximisent les gains (valeurs des objets) avec une contrainte de diversification, afin de
réduire les risques. Il est appliqué au chargement des moyens de transport mais aussi aux problèmes
de découpe.

Description
    Considérons le problème dans sa version originale. Nous souhaitons calculer quelle est la composition
des objets qui permet de maximiser un gain donné dans un sac à capacité limitée. Pour ce faire, un
algorithme génétique a été développé (voir le fichier GAexemple_Knapsack.py.py). Il est défini de la
manière suivante :
      1- Population : La population est une matrice binaire où chaque ligne est considérée comme
      solution au problème (individu). Lorsqu’un élément de la matrice dans la ligne i et dans la
      colonne j est égal à 1, ceci signifie que la ième solution pose le jème objet dans le sac.
      2- La fitness (cal_fitness) : Le codage des solutions ne garantit pas la faisabilité de la solution, à
      savoir que la somme des poids des objets sélectionnés par une solution donnée peut être
      supérieure au poids maximal. Ainsi, l’évaluation d’une solution (fitness) est égale la somme
      des valeurs des objets considérés par la solution, seulement si le poids total sera inférieur à la
      limite physique du sac. Dans le cas contraire, la valeur fitness de la solution est considérée
      nulle.
      3- Sélection : Elle consiste à prendre seulement les meilleures solutions.
      4- Croisement : Il consiste à choisir parmi les solutions sélectionnées deux parents pour générer
      une solution. Les deux parents peuvent engendrer un enfant avec une certaine probabilité
      (taux_croisement). L’enfant hérite de la moitié droite du parent 1 et de la moitié gauche du
      parent2.
      5- Mutation : Elle est appliquée aux enfants pour générer un mutant. Chaque enfant a une chance
      d’être muté (taux_mutation). La mutation consiste à inverser un bit de la solution.
      En plus de l’algorithme génétique, le fichier GAexemple_Knapsack.py génère automatiquement des
      données aléatoires du problème de sac à dos.
   
Questions
1- Faites varier le nombre d’objets, leurs poids, la capacité maximale du sac et le nombre de
générations. Que constatez-vous ? Expliquez les causes.
2- Apporter les modifications nécessaires pour aboutir seulement à des solutions faisables, si
elles existent. Voici les approches qu’il faut coder et comparer :
a. Mettre une fitness négative proportionnelle au dépassement de la capacité du sac
b. Vérifier lors de la génération des solutions, du croisement et de la mutation que ces
solutions respectent la capacité du sac.
Pour comparer les deux approches, générer 4 instances du problème et stocker les dans des
fichiers csv. Comparer les deux approches en termes de résultats et de temps de calcul.
3- Nous souhaitons utiliser cet algorithme à la gestion du portefeuille. Pour des raisons de
minimisation du risque, nous considérons que le client peut acheter plusieurs actions du même
titre dans la mesure où il respecte la limite des 20% du budget total. L’achat de plusieurs
actions ne concerne qu’un seul titre. A la suite des estimations des gains escomptés de chaque
titre, il souhaite augmenter ses gains en respectant les limites de son budget et la contrainte
de minimisation de risque. Adapter le programme à la gestion du portefeuille.
4- Choisir 4 instances des données du problème et stocker les dans des fichiers csv. Améliorer les
performances de l’algorithme génétique sur ces quatre instances. Décrivez dans le rapport les
améliorations réalisées et les motivations.
