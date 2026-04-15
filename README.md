# REDDI: A Riemannian Ensemble Learning Framework for Interpretable Differential Diagnosis of Neurodegenerative Diseases
---
This repository contains the code and supporting documents associated with the following manuscript:

- Mario Roca, Giovanni Messuti, Dmytro Klepachevskyi, Marianna Angiolelli, Simona Bonavita, Francesca Trojsi, Matteo Demuru, Emahnuel Troisi Lopez, Sylvain Chevallier, Florian Yger, Ausra Saudargienė, Pierpaolo Sorrentino, Marie-Constance Corsi (2026). REDDI: A Riemannian Ensemble Learning Framework for Interpretable Differential Diagnosis of Neurodegenerative Diseases. medXiv. [https://www.medrxiv.org/content/10.64898/2026.04.10.26350617v1](https://www.medrxiv.org/content/10.64898/2026.04.10.26350617v1)

Please cite as:

Mario Roca, Giovanni Messuti, Dmytro Klepachevskyi, Marianna Angiolelli, Simona Bonavita, Francesca Trojsi, Matteo Demuru, Emahnuel Troisi Lopez, Sylvain Chevallier, Florian Yger, Ausra Saudargienė, Pierpaolo Sorrentino, Marie-Constance Corsi (2026). REDDI: A Riemannian Ensemble Learning Framework for Interpretable Differential Diagnosis of Neurodegenerative Diseases. medXiv. [https://www.medrxiv.org/content/10.64898/2026.04.10.26350617v1](https://www.medrxiv.org/content/10.64898/2026.04.10.26350617v1)


---
## Authors:
* Mario Roca, Master student, NERV team-project, Inria Paris, Paris Brain Institute
* Giovanni Messuti, PhD student, NERV team-project, Inria Paris, Paris Brain Institute & University of Salerno, Department of Physics “E.R. Caianiello”, Fisciano, Italy
* Dmytro Klepachevskyi, Master student, Neuroscience Institute, Lithuanian University of Health Sciences, Kaunas, Lithuania
* Marianna Angiolelli, Postdoctoral researcher, Institut de Neurosciences des Systèmes, Aix-Marseille Université, Marseille, France & Università degli Studi di Napoli Parthenope, Dipartimento delle Scienze Mediche, Motorie e
del Benessere, Napoli, Italy
* Simona Bonavita, Neurologist, University of Campania “Luigi Vanvitelli”, Department of Advanced Medical and Surgical Sciences, Naples, Italy
* Francesca Trojsi, Neurologist, University of Campania “Luigi Vanvitelli”, Department of Advanced Medical and Surgical Sciences, Naples, Italy
* Matteo Demuru, Postdoctoral researcher, Università degli Studi di Napoli Parthenope, Dipartimento delle Scienze Mediche, Motorie e del Benessere, Napoli, Italy
* Emahnuel Troisi Lopez, Associate professor, Department of Education and Sport Sciences, Pegaso University, 80143 Naples, Italy
* [Sylvain Chevallier](https://sylvchev.github.io), Professor, LISN, Paris-Saclay University
* [Florian Yger](http://www.yger.fr), Associate professor, LAMSADE, Paris-Dauphine University
* Ausra Saudargienė, Professor, Neuroscience Institute, Lithuanian University of Health Sciences, Kaunas, Lithuania
* [Pierpaolo Sorrentino](https://scholar.google.com/citations?user=T1k8qBsAAAAJ&hl=en), Associate professor, Institut de Neurosciences des Systèmes, Aix-Marseille Université, Marseille, France; Università degli Studi di Napoli Parthenope, Dipartimento delle Scienze Mediche, Motorie e
del Benessere, Napoli, Italy; Institute of Applied Sciences and Intelligent Systems, National Research Council, Pozzuoli, Italy
* [Marie-Constance Corsi](https://marieconstance-corsi.netlify.app/), Researcher, NERV team-project, Inria Paris, Paris Brain Institute


---
## Abstract
Neurodegenerative diseases such as Mild Cognitive Impairment (MCI), Multiple Sclerosis (MS), Parkinson s Disease (PD), and Amyotrophic Lateral Sclerosis (ALS) are becoming more prevalent. Each of these diseases, despite its specific pathophysiological mechanisms, leads to widespread reorganization of brain activity. However, the corresponding neurophysiological signatures of these changes have been elusive. As a consequence, to date, it is not possible to effectively distinguish these diseases from neurophysiological data alone. This work uses Magnetoencephalography (MEG) resting-state data, combined with interpretable machine learning techniques, to support differential diagnosis. We expand on previous work and design a Riemannian geometry-based classification pipeline. The pipeline is fed with typical connectivity metrics, such as covariance or correlation matrices. To maintain interpretability while reducing feature dimensionality, we introduce a classifier-independent feature selection procedure that uses effect sizes derived from the Kruskal-Wallis test. The ensemble classification pipeline, called REDDI, achieved a mean balanced accuracy of 0.81 (+/-0.04) across five folds, representing a 13% improvement over the state-of-the-art, while remaining clinically transparent. As such, our approach achieves reliable, interpretable, data-driven, operator-independent decision-support tools in neurology.

---
## Code

This folder contains all the scripts used to run the analyses and to generate the figures presented in the project.

To install all required Python packages, run:

`pip install -r requirements.txt`

The environment has been tested with **Python 3.11.9**.

The `Code` directory includes three subfolders:

### **• Code_for_plots/**
Contains the scripts used to reproduce the figures shown in the paper.  
By default, these scripts load the precomputed results stored in the `Results/` directory.

### **• Train_models/**
Includes the scripts used to train the classifiers for each feature type (**ATM**, **covariance matrices**, **correlation matrices**, **PSD**).  
Each subfolder corresponds to a different classifier implementation:

- LDA  
- SVM  
- XGBoost  
- Riemannian models  
- Neural Networks (NN)

Running these scripts generates new results, which are saved in a **newly created subfolder** inside `Results/` so that the original reference outputs are not overwritten.

### **• _libs/**
Contains internal modules (utility functions, helper classes, and shared components) used throughout the project.  
These files define reusable functionality imported by both the training and plotting scripts.

---

## Results

This folder contains the outputs of the model training procedures.  
When running scripts from `Code/Train_models/`, new results are saved in a **new subdirectory** inside `Results/` rather than overwriting the existing ones.

---

## Paper_Figures

This folder contains all the images included in the paper.

---

## data

This folder contains the datasets used to train the models.  
The scripts in `Code/Train_models/` load data from this directory by default.



