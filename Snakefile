ids = ['JZ2', 'JZ5']

train_test_sce = expand('data/jz/{id}/sce_train_test.rds', id = ids)

rule all:
    input:
        train_test_sce
        
rule make_train_test_sce:
    input:
        input_sce = 'data/jz/{id}/sce_qc.rds' # after cell filtering
    params:
        reporter_gene = 'dsRED',
        train_prop = 0.7,
        topHVGs = 500,
        seed = 42, 
        sort_genes = 0
    output:
        output_sce = 'data/jz/{id}/sce_train_test.rds'
    threads: 8 # if use snakemake --cores 16 can run 2 jobs in parallel 
    shell:
        "Rscript scripts/prep_train_test.R \
         --path_to_sce {input.input_sce} \
         --reporter_gene {params.reporter_gene} \
         --train_prop {params.train_prop} \
         --topHVGs {params.topHVGs} \
         --seed {params.seed} \
         --sort_genes {params.sort_genes} \
         --path_to_train_test_sce {output.output_sce}"
    