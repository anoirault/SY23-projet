# SY23 : PROJET - Implémentation d'algorithme d'Apprentissage par Renforcement

Le but de ce projet et d'implémenter et de tester différents algroithmes d'apprentissage par renforcement (REINFORCE et Proximal Policy Optimisation (PPO)) et de les tester sur différents environnements.

Les environnements de test sont *InvertedPendulul-v4* et *InvertedDoublePendulum-v4* de la librairie `gymnasium` d'OpenAI. Dans ces deux environnements, l'objectif est de maintenir un ou plusieur pendules, à la verticale, pour ce faire, une récompense positive est retourné tant que le pendule est verticale.

L'apprentissage par renforcement est une famille d'algorithme permettant d'apprendre un comportement en maximisant une somme de récompense.

La plupart des algorithmes essayent d'apprendre la *Q function* aussi appelé la *state-action function* et d'en déduire une politic d'action ou *policy* optimal, cependant, REINFORCE et PPO sont des algorithmes de recherche direct. C'est algorithme vont approximer directement une *policy* en effectuant une montée de gradient sur la somme des récompenses. La particularité de PPO est d'ajouté un *actor-critic* permettant d'évaluer la qualité de l'action prise par l'agent.

Ce projet contient une implémentation de l'algorithme REINFORCE (`reinforce.ipynb`) avec les poids entrainer pour les deux environnements, une implémentation de PPO (`ppo.ipynb`) inspiré de l'article original publié par [Schulman et. alii.](https://arxiv.org/pdf/1707.06347.pdf) ainsi qu'une comparaison avec l'implémentation disponible dans la [librairie pytorch](https://pytorch.org/rl/tutorials/coding_ppo.html).