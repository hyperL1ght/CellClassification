suppressPackageStartupMessages({
  library(dplyr)
  library(tibble) #rownames_to_column 
  library(SingleCellExperiment) # some functions with same names as in dplyr
  library(scater)
  library(scran)
  library(caret)
  library(argparse)
})

parser <- ArgumentParser(description = "Split to train/test sce, normalize separately & HVGs by train")
parser$add_argument('--path_to_sce', metavar='DIRECTORY', type='character',
                    help="Path to sce")
parser$add_argument('--reporter_gene', metavar='gene_symbol', type='character',
                    help="name of reporter gene")
parser$add_argument('--train_prop', metavar='value between [0., 1.]', type='double',
                    help="proportion of cells in train")
parser$add_argument('--topHVGs', metavar='integer', type='integer',
                    help="Number of top HVGs to keep in downstream steps")
parser$add_argument('--seed', metavar='integer', type='integer',
                    help="seed for random split")
parser$add_argument('--sort_genes', metavar='integer', type='integer',
                    help="whether to sort genes")
parser$add_argument('--path_to_train_test_sce', metavar='DIRECTORY', type='character',
                    help="Path to output sce")
args <- parser$parse_args()

prep_train_test <- function(path_to_sce, reporter_gene, train_prop, 
                            seed, topHVGs, sort_genes, path_to_train_test_sce){
  
  sce <- readRDS(path_to_sce)
  sce$pos <- counts(sce)[reporter_gene, ] > 0
  sce$reporter_gene_counts <- counts(sce)[reporter_gene, ]
  sce <- sce[rownames(sce) != reporter_gene, ] # REMOVE reporter_gene 
  
  # adding chromosome locations for gene sorting (for use with Unet) 
  if(sort_genes){
      #retrieve gene annotations from BioMart database 
      sce <- getBMFeatureAnnos(sce, ids = rowData(sce)$ID, filters = "ensembl_gene_id",
                           attributes = c("ensembl_gene_id", "chromosome_name", "start_position", "end_position"),
                           dataset = "hsapiens_gene_ensembl")
      # sorting by chromosome_name and start_position
      rowData(sce)$chromosome_name <- factor(rowData(sce)$chromosome_name, levels = c(c(1:22), 'X', 'Y', 'MT'))
      tmp_df <- as.data.frame(rowData(sce)) %>% 
              rownames_to_column('rownames') %>% 
              dplyr::select(rownames, chromosome_name, start_position, end_position) %>%
              dplyr::filter(!is.na(chromosome_name)) %>%
              dplyr::arrange(chromosome_name, start_position)
      
      # append gene order to sce
      sce@metadata$gene_order <- tmp_df$rownames
      
      # sort genes   
      sce <- sce[tmp_df$rownames, ]
      
  }
  
  set.seed(seed)
  sce <- sce[, sample(colnames(sce), replace = FALSE)] # shuffle cells
  train_ix <- createDataPartition(as.character(sce$pos), times = 1, p = train_prop) # stratified random split 
  sce_list <- list(train = sce[, train_ix$Resample1], test = sce[, -train_ix$Resample1])
  sce_list[['train']]$split <- 'train'
  sce_list[['test']]$split <- 'test'
  
  normalize_sce <- function(sce){
    
    print('Normalize input sce ...')
    num_cells <- dim(sce)[2]
    min_size <- min(100, floor(dim(sce)[2] * 0.3))
    max_win <- min(101, min_size + 1)
    
    clusters <- quickCluster(sce, min.size=min_size, min.mean= 0.1, method="igraph", use.ranks=TRUE)
    sce <- computeSumFactors(sce, sizes=seq(21, max_win, 5), min.mean= 0.1, clusters=clusters)
    if(min(sizeFactors(sce)) < 0) stop('Negative size factors exist')
    sce <- normalize(sce)
    
    return(sce)
    
  }
  
  # normalize 
  sce_list <- lapply(sce_list, normalize_sce)
  
  # hvg using train only
  var_fit <- trendVar(sce_list[['train']], assay.type="logcounts", use.spikes=FALSE)
  var_out <- decomposeVar(sce_list[['train']], fit = var_fit, assay.type="logcounts")  
  var_out_sig <- var_out[var_out$FDR < 0.05 & var_out$bio > 0, ]
  var_out_sig <- var_out_sig[order(var_out_sig$bio, decreasing = TRUE),] 
  var_out_not_sig <- var_out[!rownames(var_out) %in% rownames(var_out_sig), ]
  var_out_not_sig <- var_out_not_sig[order(var_out_not_sig$bio, decreasing = TRUE),] 
  hvg_order <- c(rownames(var_out_sig), rownames(var_out_not_sig))                                                                 
  # add hvg metadata
  sce <- cbind(sce_list[['train']], sce_list[['test']])
  sce@metadata$hvg_order <- hvg_order
  sce@metadata$hvg_sig <- rownames(var_out_sig)
  top_genes <- rownames(var_out_sig)[1:topHVGs]
  rowData(sce)$is_hvgs <- rownames(sce) %in% top_genes
  
  saveRDS(sce, file = path_to_train_test_sce)
  
}

prep_train_test(path_to_sce = args$path_to_sce, reporter_gene = args$reporter_gene, train_prop = args$train_prop, 
                topHVGs = args$topHVGs, seed = args$seed, sort_genes = args$sort_genes,
                path_to_train_test_sce = args$path_to_train_test_sce)









