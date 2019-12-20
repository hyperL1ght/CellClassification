from functools import partial
import pandas as pd
from tqdm import tqdm_notebook 
from sklearn.model_selection import cross_validate, StratifiedKFold, StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import make_scorer, fbeta_score, accuracy_score, roc_auc_score, average_precision_score, confusion_matrix

def make_models(seed = 0):
    transformer = Pipeline(steps = [('scaler', StandardScaler()),
                                ('pca', PCA(n_components=100))])
    LR = Pipeline(steps = [('preprocessing', transformer), 
                           ('classifier', LogisticRegression(class_weight = 'balanced', solver = 'newton-cg'))])
    GNB = Pipeline(steps = [('preprocessing', transformer), 
                            ('classifier', GaussianNB())])
    # base_estimator must support for sample weighting in fit for AdaBoost
    AdaGNB = Pipeline(steps = [('preprocessing', transformer),
                               ('classifier', AdaBoostClassifier(base_estimator = GaussianNB(), 
                                                                 n_estimators=50, random_state=seed))])
    SVM = Pipeline(steps = [('preprocessing', transformer), 
                            ('classifier', LinearSVC(class_weight='balanced', dual=False, max_iter=5000, random_state=seed))])
    MLP = Pipeline(steps = [('preprocessing', transformer), 
                            ('classifier', MLPClassifier(hidden_layer_sizes=(5, 16, 8), learning_rate='adaptive',
                                                         max_iter=1000, random_state = seed, activation='relu'))])
    bagMLP = Pipeline(steps = [('classifier', BaggingClassifier(base_estimator=MLP, n_estimators=50, bootstrap=True, 
                                                            max_samples=1.0, n_jobs=-1, 
                                                            random_state=seed, oob_score=True))])
    RF = Pipeline(steps = [('classifier', RandomForestClassifier(n_estimators=500, n_jobs=-1, class_weight = 'balanced',
                                                             random_state=seed, oob_score = True))])
    models = {}
    models['LR'] = {'cv': True, 'bagging': False, 'model': LR}
    models['GNB'] = {'cv': True, 'bagging': False, 'model': GNB}
    models['AdaGNB'] = {'cv': True, 'bagging': False, 'model': AdaGNB}
    models['SVM'] = {'cv': True, 'bagging': False, 'model': SVM}
    models['MLP'] = {'cv': True, 'bagging': False, 'model': MLP}
    models['Bagging MLP'] = {'cv': True, 'bagging': True, 'model': bagMLP}
    models['RF'] = {'cv': True, 'bagging': True, 'model': RF}
    
    return models 

def make_score(metric, argument_dict = {}, needs_threshold = False):
    return {'callable': partial(metric, **argument_dict), 'needs_threshold': needs_threshold}

def build_models(models, X_train, X_test, y_train, y_test, cv, cv_random_state, cv_scoring, eval_scoring, eid, verbose = False):
    cv_scores = {}
    for model_name, model in tqdm_notebook(models.items(), desc="Building models"):
        if(model['cv']):
            if verbose: print('Cross-validation for', model_name)
            cv_scores[model_name] = cross_validate(model['model'], X_train, y_train, 
                                                   cv = StratifiedKFold(n_splits = cv, shuffle=True, 
                                                                        random_state=cv_random_state), 
                                                   scoring = cv_scoring, n_jobs=-1)
        if verbose: print('Fitting', model_name)
        models[model_name]['model'] = model['model'].fit(X_train, y_train)
    
    # collect cv scores     
    cv_rows = []
    for ok, ov in cv_scores.items():
        for ik, iv in ov.items(): 
            if('test_' in ik):
                row = {'model': ok, 'score_type': ik, 'mean': iv.mean(), 'std': iv.std()}
                cv_rows.append(row)
    cv_scores_df = pd.DataFrame(cv_rows) 
    cv_scores_df['id'] = eid
    
    # collect evaluation scores
    eval_rows = []
    for model_name, model in tqdm_notebook(models.items(), desc="Evaluating models"):
        for metric_name, metric in eval_scoring.items():
            if metric['needs_threshold']:
                if hasattr(model['model']['classifier'], 'decision_function'):
                # decision function
                    rows = [{'model': model_name, 'split': x, 'score_type': metric_name, \
                             'score': metric['callable'](y, model['model'].decision_function(z))} \
                            for x, y, z in zip(['train', 'test'], [y_train, y_test], [X_train, X_test])]
                    eval_rows.extend(rows)
                elif hasattr(model['model']['classifier'], 'predict_proba'):
                # probability of the positive class 
                    rows = [{'model': model_name, 'split': x, 'score_type': metric_name, \
                             'score': metric['callable'](y, model['model'].predict_proba(z)[:, 1])} \
                            for x, y, z in zip(['train', 'test'], [y_train, y_test], [X_train, X_test])]
                    eval_rows.extend(rows)
                else:
                    raise ValueError('Estimator must have either predict_proba or decision_function method')
            else:
                # discrete predictions
                rows = [{'model': model_name, 'split': x, 'score_type': metric_name, \
                         'score': metric['callable'](y, model['model'].predict(z))} \
                        for x, y, z in zip(['train', 'test'], [y_train, y_test], [X_train, X_test])]
                eval_rows.extend(rows)
    
    eval_scores_df = pd.DataFrame(eval_rows) 
    eval_scores_df['id'] = eid
        
    return cv_scores_df, eval_scores_df

def array_to_df(arr, colname_prefix = 'gene', colnames = None, rownames = None):
    if colnames is None:
        colnames = [colname_prefix+str(i+1) for i in range(arr.shape[1])]
    if rownames is None:
        rownames = [i for i in range(arr.shape[0])]
        
    df=pd.DataFrame(data=arr[0:,0:],
                index=rownames,
                columns=colnames)
    return df

def prepare_fastai_inputs(sce, assay, gene_filter = True, val_pro = 0.3, seed = 0):
    
    if gene_filter: sce = sce[:, sce.var['is_hvgs'] == 1]
    sce_train = sce[sce.obs['split'] == 'train', ]
    sce_test = sce[sce.obs['split'] == 'test', ]
    
    if assay == 'raw':
        X_train = sce_train.X.todense()
        X_test = sce_test.X.todense()
    elif assay == 'normalized':
        X_train = sce_train.layers['logcounts'].todense()
        X_test = sce_test.layers['logcounts'].todense()
    
    y_train = sce_train.obs['pos']
    y_test = sce_test.obs['pos']
    
    train_df = array_to_df(X_train, colnames = sce.var[(sce.var['is_hvgs'] == 1)].index, rownames = y_train.index)
    train_df['label']= y_train
    test_df = array_to_df(X_test, colnames = sce.var[(sce.var['is_hvgs'] == 1)].index, rownames = y_test.index)
    test_df['label']= y_test
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_pro, random_state=seed)
    X, y = train_df[train_df.columns.difference(['label'])], train_df['label']
    sss.get_n_splits(X, y)
    _, val_idx = next(sss.split(X, y))
    
    return train_df, test_df, val_idx





    
