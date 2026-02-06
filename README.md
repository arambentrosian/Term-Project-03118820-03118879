# graph-emb1
Big data management and information systems - Semester project

First run these commands to setup packages:

    python3 -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.8.0+cpu.html

I.Classification
    a. Unsupervised methods

     python experiments/mutag_graph2vec.py
     python experiments/enzymes_graph2vec.py
     python experiments/imdb_multi_graph2vec.py 
     
     python experiments/mutag_netlsd.py
     python experiments/enzymes_netlsd.py
     python experiments/imdb_multi_netlsd.py

    b. GIN

     python experiments/mutag_gin.py
     python experiments/enzymes_gin.py
     python experiments/imdb_multi_gin.py

II. Clustering
    a. Unsupervised methods

     python experiments/run_kmeans_clustering.py
    
    b. GIN

     python experiments/run_gin_clustering.py

III. Perturbations
    a. Unsupervised methods

     python experiments/run_embedding_stability.py

    b. GIN

     python experiments/run_gin_stability.py