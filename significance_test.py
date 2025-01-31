from sklearn import metrics
import torch
import math
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from src.animation_utils import compute_auroc
from scipy import stats  # Import the 'stats' module from SciPy.
import os


def get_files(dataset_n, model_n, file = '/'):
    """
    Get the files associated to each best performance of each model in each dataset.

    :param dataset_n: dataset name.
    :param model_n: model name.
    :return: Logits of the positives and negatives edges, together with the AUROC for each split and seed.
    """

    path = './performance' + file + dataset_n + '/' + model_n + '/'
    pos_path = path + 'positives.npy'
    neg_path = path + 'negatives.npy' 
    AUROC_path = path + 'AUROC.npy' 

    pos = np.load(pos_path)
    neg = np.load(neg_path)
    AUROC = np.load(AUROC_path)

    return pos, neg, AUROC

def get_accuracies(positives_tensor, negatives_tensor):
    """
    Convert a set of predicted logits to a respective accuracy for the significance tests. This is done also for a reduced set.

    :param positives_tensor: Logits associated with the positive edges that have been predicted.
    :param negatives_tensor: Logits associated with the negative edges that have been predicted.
    :return: output: a Tensor where each element is the respective accuracy, after having applied the optimal threshold of the AUROC to the input positives + negatives
    :return: auroc: The respective auroc for comparison purposes.
    """
    # Combine the positive and negative tensors
    all_tensor = np.concatenate((positives_tensor, negatives_tensor))

    # Create the corresponding labels (1 for positives, 0 for negatives)
    labels = np.concatenate((np.ones(positives_tensor.shape[0]), np.zeros(negatives_tensor.shape[0])))

    # Calculate the AUROC using sklearn's roc_auc_score function
    auroc = metrics.roc_auc_score(labels, all_tensor)

    fpr, tpr, thresholds = metrics.roc_curve(labels, all_tensor, pos_label = 1)

    # print(thresholds)
    roc_auc = metrics.auc(fpr, tpr)

    assert auroc == roc_auc
    
    gmeans = np.sqrt(tpr * (1-fpr))
    # locate the index of the largest g-mean
    ix = np.argmax(gmeans)
 
    optimum_threshold = thresholds[ix]

    # print("The optimum Threshold is: ", optimum_threshold)
    # print("True Positive Rate is: ", tpr[ix])
    # print("False Positive Rate is: ", fpr[ix])

    output_1 = positives_tensor >= optimum_threshold
    output_2 = negatives_tensor < optimum_threshold


    output_1 = output_1.astype(int)
    output_2 = output_2.astype(int)

    output = np.concatenate((output_1, output_2))


    return output, np.round(roc_auc, 4)


def get_new_tensor(pos_tensor, neg_tensor, AUROCs):
    """
    Convert a set of predicted logits to a respective accuracy for the significance tests. This is done also for a reduced set.

    :param pos_tensor: All the Logits associated with the positive edges that have been predicted.
    :param neg_tensor: All the Logits associated with the negative edges that have been predicted.
    :param AUROCs: Array of the aurocs associated to each seed and data split.
    :return: output_accuracy: Accuracy tensor of the whole set of elements to predict 
    """
    batch_size = pos_tensor.shape[0] // AUROCs.shape[0]

    output_accuracy = np.array([])
    i = 0
    for batch_idx in range(0, pos_tensor.shape[0], batch_size):
     
        current_auroc = AUROCs[i]
        intermediate_accuracy, auroc = get_accuracies(pos_tensor[batch_idx: batch_idx + batch_size], neg_tensor[batch_idx: batch_idx + batch_size])

        assert current_auroc == auroc

        i+=1

        output_accuracy = np.concatenate((output_accuracy, intermediate_accuracy))

    assert 2*pos_tensor.shape[0] == output_accuracy.shape[0]


    # print(np.sum(output_accuracy)/output_accuracy.shape[0])

    return output_accuracy


def print_test_results(statistic, p_value):
    """
    Print test results including the test statistic and p-value.

    :param statistic: Test statistic.
    :param p_value: P-value.
    """
    print(f"\t\t\tTest statistic: {statistic}")
    print(f"\t\t\tP-value: {p_value}")

def check_significance(p_value, alphas=[0.01, 0.05, 0.1]):
    """
    Check the significance of a test based on the given p-value.

    :param p_value: P-value.
    :param alphas: List of significance levels to check against (default is [0.01, 0.05, 0.1]).
    :return: The significance level at which the test is significant or None if not significant.
    """
    for alpha in alphas:
        significant = p_value < alpha
        if significant:
            break
    print(f"\t\t\t--> The test is{' NOT' if not significant else ''} significant at {alpha} level")
    return alpha if significant else None

def shapiro_test(sample, alphas=[0.01, 0.05, 0.1]):
    """
    Perform the Shapiro-Wilk test for normality on a sample.

    :param sample: Sample data to be tested for normality.
    :param alphas: List of significance levels to check against (default is [0.01, 0.05, 0.1]).
    :return: The significance level at which the test is significant or None if not significant.
    """
    statistic, p_value = stats.shapiro(sample)
    print(f"\t\tShapiro-Wilk test for normality:")
    print_test_results(statistic, p_value)
    significance = check_significance(p_value, alphas)
    print()
    return significance

def t_test(sample1, sample2=None, pop_mean=0, alternative="greater", alphas=[0.01, 0.05, 0.1]):
    """
    Calculate a T-test for the mean of one group of scores or two related samples.

    :param sample1: First sample data.
    :param sample2: Second sample data (for two related samples).
    :param pop_mean: Population mean (for one group of scores).
    :param alternative: Alternative hypothesis for the test ('greater', 'less', or 'two-sided').
    :param alphas: List of significance levels to check against (default is [0.01, 0.05, 0.1]).
    :return: The significance level at which the test is significant or None if not significant.
    """
    if sample2 is None:
        print(f"\t\tT-test for the mean of ONE group of scores with population mean {pop_mean} and {alternative} alternative:")
        statistic, p_value = stats.ttest_1samp(sample1, pop_mean=pop_mean, alternative=alternative)
    else:
        print(f"\t\tT-test for the mean of TWO RELATED samples of scores with {alternative} alternative:")
        statistic, p_value = stats.ttest_rel(sample1, sample2, alternative=alternative)
    print_test_results(statistic, p_value)
    significance = check_significance(p_value, alphas)
    print()
    return significance

def wilcoxon_test(sample1, sample2=None, alternative="greater", alphas=[0.01, 0.05, 0.1], zero_method=["wilcox", "pratt", "zsplit"]):
    """
    Calculate the Wilcoxon signed-rank test.

    :param sample1: First sample data.
    :param sample2: Second sample data (for two related samples).
    :param alternative: Alternative hypothesis for the test ('greater', 'less', or 'two-sided').
    :param alphas: List of significance levels to check against (default is [0.01, 0.05, 0.1]).
    :param zero_method: Method to handle zero differences ('wilcox', 'pratt', or 'zsplit').
    :return: List of significance levels at which the test is significant or None if not significant for each zero_method.
    """
    significance = []
    if isinstance(zero_method, str):
        zero_method = [zero_method]
    for zero_method in zero_method:
        print(f"\t\tWilcoxon signed-rank test with {zero_method} method and {alternative} alternative:")
        statistic, p_value = stats.wilcoxon(sample1, sample2, zero_method=zero_method, correction=False, alternative=alternative)
        print_test_results(statistic, p_value)
        significance.append(check_significance(p_value, alphas))
    print()
    return significance

def compute_significance(dataset_n, mod1, mod2, metric = 'accuracy', file = ''):
    """
    Compute the significance for a pair of algorithm over a dataset.

    :param dataset_n: dataset name.
    :param mod1: The model that we want to show that is greater than the other.
    :param mod2: The null hypothesis model.
    :param metric: Do we use the aurocs or the accuracies? Alternatives : ['accuracy', 'auc']
    :return: result: significance level alpha. None if not significative.
    :return: auc_stri: Descriptive string for i model.
    """

    pos1, neg1, AUROC1 = get_files(dataset_n, mod1, file = file)
    pos2, neg2, AUROC2 = get_files(dataset_n, mod2, file = file)

    if metric == 'auc':
        metric1 = AUROC1
        metric2 = AUROC2
    elif metric == 'accuracy':
        acc1 = get_new_tensor(pos1, neg1, AUROC1)
        acc2 = get_new_tensor(pos2, neg2, AUROC2)
        metric1 = acc1
        metric2 = acc2


    auc_str1 = str(np.round(np.mean(AUROC1)*100, 2)) + ' +- ' + str(np.round(np.std(AUROC1)*100, 1))
    auc_str2 = str(np.round(np.mean(AUROC2)*100, 2)) + ' +- ' + str(np.round(np.std(AUROC2)*100, 1))

    # Refuse the null hypothesis and confirm that they are normally distributed
    if shapiro_test(metric1) == None and shapiro_test(metric2) == None and 3 > 4:
        print("BOTH the distribution are normally distributed.....")
        result = t_test(metric1, metric2)
        return result, auc_str1, auc_str2
    else:
        print("One of the distribution is not normally distributed")
        result = wilcoxon_test(metric1, metric2)
        return result, auc_str1, auc_str2



def plot_table(dataset_list, models, mod1, metric, file = ''):
    """
    Compute the significancy for a set of models against an anchor one, and repeated for a set of datasets.

    :param dataset_list: List of the scrutinized datasets.
    :param models: List of models to compare.
    :param mod1: The anchor model.
    :param metric: Do we use the aurocs or the accuracies? Alternatives : ['accuracy', 'auc']
    :return: Display the pandas dataframe with the results.
    """
    dataframe = pd.DataFrame(index = models + [mod1], columns = dataset_list)

    for dataset in dataset_list:
        if dataset in ['amazon_ratings', 'roman_empire', 'minesweeper', 'questions']:
            metric = 'accuracy'
        for model in models:
            print(f'EVALUATE {mod1} AGAINST {model} in {dataset}\n\n')
            if model == 'disenlink' and dataset in ['amazon_ratings', 'roman_empire', 'minesweeper', 'questions']:
                dataframe.loc[model, dataset] = 'OOM'
                continue
            result, auc_mod1, auc_mod2 = compute_significance(dataset, mod1, model, metric = metric, file = file)

            if result in [0.1, 0.05, 0.01]:
                dataframe.loc[model, dataset] = auc_mod2 + '_' + str(result) + '_t'
            elif isinstance(result, list):
                filtered_values = [val for val in result if val is not None]

                if filtered_values:
                    result = min(filtered_values)
                else:
                    result = None
                dataframe.loc[model, dataset] = auc_mod2 + '_' + str(result) + '_w'
                    
        dataframe.loc[mod1, dataset] = auc_mod1


    return dataframe

def plot_explainability(dataset_list, models, mod1, file, metric_name = 'exp_homo'):

    dataframe = pd.DataFrame(index = models + [mod1], columns = dataset_list)

    for model in (models + [mod1]):

        for dataset in dataset_list:

            path = 'performance' + file + dataset + '/' + model

        
            try:
                metric = np.load(path + '/' + metric_name)
            except FileNotFoundError:
                metric = [0]
        
            mean, std = np.mean(metric), np.std(metric)

            dataframe.loc[model, dataset] = str(round(mean, 4)) + ' +- ' + str(round(std, 4))

    print(dataframe)

def display_results_per_datasets(file, metric_type, models, dataset_list):
    """
    Display the mean and standard deviation of the results in the dataframe.

    :param file: Type of readout results.
    :param metric_type: Choose among 'AUROC.npy'/'exp_homo.npy'/'exp_hetero.npy'/'exp_mean.npy'
    :param models: List of model names.
    :param dataset_list: List of dataset names.
    """
    fig, ax = plt.subplots()
    bar_width = 0.1
    opacity = 0.8
    index = np.arange(len(dataset_list))
    
    for i, model in enumerate(models):
        means = []
        stds = []
        for dataset in dataset_list:
            try:
                path = './performance' + file + dataset + '/' + model + '/' + metric_type
                metric = np.load(path)
                mean = np.mean(metric)
                std = np.std(metric)
            except FileNotFoundError:
                mean = 0
                std = 0
            means.append(mean)
            stds.append(std)
    
        ax.bar(index + i * bar_width, means, bar_width, alpha=opacity, yerr=stds, label=model)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Mean')
    ax.set_title('Mean and Standard Deviation of Results')
    ax.set_xticks(index + bar_width * len(models) / 2)
    ax.set_xticklabels(dataset_list)
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    

def display_results_per_metric(file, metric_list, models, dataset, name_list):
    """
    Display the mean and standard deviation of the results in the dataframe.

    :param file: Type of readout results.
    :param metric_list: List of elements among 'AUROC.npy'/'exp_homo.npy'/'exp_hetero.npy'/'exp_mean.npy'
    :param models: List of model names.
    :param dataset_list: List of dataset names.
    """
    plt.rcParams.update({
        'font.size': 20,          # General font size
        'axes.titlesize': 20,     # Title font size
        'axes.labelsize': 18,     # X and Y axis label size
        'xtick.labelsize': 14,    # X-axis tick label size
        'ytick.labelsize': 14,    # Y-axis tick label size
        'legend.fontsize': 16,    # Legend font size
        'figure.titlesize': 24     # Overall figure title font size
    })
    fig, ax = plt.subplots()
    bar_width = 0.1
    opacity = 0.8
    index = np.arange(len(metric_list))
    
    for i, model in enumerate(models):
        means = []
        stds = []
        for metric_name in metric_list:
            try:
                path = './performance' + file + dataset + '/' + model + '/' + metric_name
                metric = np.load(path)
                mean = np.mean(metric)
                std = np.std(metric)
            except FileNotFoundError:
                mean = 0
                std = 0
            means.append(mean)
            stds.append(std)
        if model == 'GRAFF':
            model = 'GRAFF-LP'
    
        ax.bar(index + i * bar_width, means, bar_width, alpha=opacity, yerr=stds, label=model)
    
    #ax.set_xlabel(f'{dataset}')
    ax.set_ylabel('$AUC$')
    ax.set_title('')
    ax.set_xticks(index + bar_width * len(models) / 2)

    

    ax.set_xticklabels(name_list)
    
    # Move the legend to the bottom-right corner
    ax.legend(loc='lower right', bbox_to_anchor=(1, 0))
    
    plt.tight_layout()
    prefix = f'AUROCS_bar/{dataset}/'
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    pdf_name = f'{prefix}+file.pdf'
    plt.savefig(pdf_name, format='pdf', bbox_inches='tight')  # Use bbox_inches to handle text cutoff
    print(f'Figure saved as {pdf_name}')
    plt.show()


def display_differential_results(file1, file2, metric_type1, metric_type2, models, dataset_list, title='', save_as_pdf=True, pdf_name='figure.pdf'):
    """
    Display the difference between two bar charts representing the mean and standard deviation of the results.
    
    :param file1: Type of readout results at time 1.
    :param file2: Type of readout results at time 2.
    :param metric_type1: Choose among 'AUROC.npy'/'exp_homo.npy'/'exp_hetero.npy'/'exp_mean.npy'
    :param metric_type2: Choose among 'AUROC.npy'/'exp_homo.npy'/'exp_hetero.npy'/'exp_mean.npy'
    :param models: List of model names.
    :param dataset_list: List of dataset names.
    :param title: Title of the plot.
    :param save_as_pdf: Boolean, if True saves the plot as a PDF.
    :param pdf_name: Name of the output PDF file.
    """
    
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust figure size for better readability
    bar_width = 0.1
    opacity = 0.8
    index = np.arange(len(dataset_list))

    for i, model in enumerate(models):
        means1 = []
        means2 = []
        for dataset in dataset_list:
            try:
                path1 = './performance' + file1 + dataset + '/' + model + '/' + metric_type1
                metric1 = np.load(path1)
                mean1 = np.round(np.mean(metric1), 2)
            except FileNotFoundError:
                mean1 = 0
            means1.append(mean1)
            
            try:
                path2 = './performance' + file2 + dataset + '/' + model + '/' + metric_type2
                metric2 = np.load(path2)
                mean2 = np.round(np.mean(metric2), 2)
            except FileNotFoundError:
                mean2 = 0
            means2.append(mean2)
        
        improvement = [mean2 - mean1 for mean1, mean2 in zip(means1, means2)]
        minimum = [min(mean2, mean1) for mean1, mean2 in zip(means1, means2)]
        maximum = [max(mean2, mean1) for mean1, mean2 in zip(means1, means2)]

        # Percentage improvement
        changes = [f'{round(100*((mean2 - mean1) / mean1), 2)}%' if mean1 != 0 else '0%' for mean1, mean2 in zip(means1, means2)]
        
        colors = ['green' if imp > 0 else 'red' for imp in improvement]

        # Plot maximum and minimum bars
        ax.bar(index + (i+0.1) * bar_width, maximum, bar_width, alpha=opacity, color=colors, edgecolor='black')
        ax.bar(index + (i+0.1) * bar_width, minimum, bar_width, alpha=opacity, label=model, edgecolor='black')

        #Adjust position of text
        for j, value in enumerate(changes):
            ax.text(index[j] + (i+0.1) * bar_width, maximum[j] + 0.03, value, ha='center', va='bottom', color='black', fontsize=9)  # Smaller font size, adjusted position

    # Set labels, title, and ticks
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Auroc Improvement', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(index + bar_width * len(models) / 2)
    ax.set_xticklabels(dataset_list, fontsize=12)
    ax.legend()

    # Make sure the layout is tight
    plt.tight_layout()


    # Save as PDF if save_as_pdf is True
    if save_as_pdf:
        pdf_name = f'{title}.pdf'
        plt.savefig(pdf_name, format='pdf', bbox_inches='tight')  # Use bbox_inches to handle text cutoff
        print(f'Figure saved as {pdf_name}')

    plt.show(block=False)

def compare_among_categories(category_elements, model_list, dataset_list):

    for i, cat_item in enumerate(category_elements[1:]):

        print(f'Comparison of {cat_item[:-4]}')


        display_differential_results(file_gradient, file_gradient, category_elements[0], cat_item, model_list, dataset_list, title = cat_item[:-4] + ' gradient')
        display_differential_results(file_hadamard, file_hadamard, category_elements[0], cat_item, model_list, dataset_list, title = cat_item[:-4] + ' hadamard')
        input('')


def compare_among_readouts(category1, category2, file1, file2, model_list, dataset_list):

    for i, (cat1_item, cat2_item) in enumerate(zip(category1, category2)):

        print(f'Comparison of {cat1_item[:-4]} and {cat2_item[:-4]}')

        display_differential_results(file1, file2, cat1_item, cat1_item, model_list, dataset_list, title = cat1_item[:-4])
        display_differential_results(file1, file2, cat2_item, cat2_item, model_list, dataset_list, title = cat2_item[:-4])

        print(f'Table of {cat1_item[:-4]} Hadamard \n')
        plot_explainability(dataset_list, model_list[:-1], model_list[-1], file1, metric_name = cat1_item)
        print(f'\n Table of {cat1_item[:-4]} Gradient \n')
        plot_explainability(dataset_list, model_list[:-1], model_list[-1], file2, metric_name = cat1_item)

        print(f'\n Table of {cat2_item[:-4]} Hadamard \n')
        plot_explainability(dataset_list, model_list[:-1], model_list[-1], file1, metric_name = cat2_item)
        print(f'\n Table of {cat2_item[:-4]} Gradient \n')
        plot_explainability(dataset_list, model_list[:-1], model_list[-1], file2, metric_name = cat2_item)

        input('')
def plot_box_plot(model_name, dataset_name):
    # Step 1: Load the distributions
    neg_distribution = np.load(f"gradient_distributions/{model_name}/{dataset_name}/neg_distribution.npy")
    pos_distribution = np.load(f"gradient_distributions/{model_name}/{dataset_name}/pos_distribution.npy")

    # Step 2: Ensure that the shapes of both distributions are the same
    assert neg_distribution.shape == pos_distribution.shape, "Both distributions must have the same shape"

    n, n_samples = neg_distribution.shape

    # Step 3: Create a figure and axis for plotting
    fig, ax = plt.subplots(figsize=(12, 6))

    # Step 4: Prepare to plot the box plots for each distribution along the 'n' dimension
    for i in range(n):
        # For each row (distribution), extract samples from positive and negative distributions
        pos_data = pos_distribution[i, :]
        neg_data = neg_distribution[i, :]

        # Compute AUROC (replace with your real function)
        auroc = round(100 * compute_auroc(torch.from_numpy(neg_data), torch.from_numpy(pos_data)), 2)

        # Create a box plot for the positive distribution
        pos_bp = ax.boxplot(pos_data, positions=[2 * i], widths=0.6, patch_artist=True, 
                            boxprops=dict(facecolor='blue', color='blue'))

        # Create a box plot for the negative distribution
        neg_bp = ax.boxplot(neg_data, positions=[2 * i + 1], widths=0.6, patch_artist=True, 
                            boxprops=dict(facecolor='red', color='red'))

        # Step 4b: Add text for GS = auroc above each layer (above the associated box plot)
        offset = 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0])  # 5% of the y-axis range

        # Adjust the position of the text, moving it upwards
        ax.text(2 * i + 0.5, ax.get_ylim()[1] + offset, f'$GS = {auroc:.2f}$', 
                ha='center', va='bottom', fontsize=12, color='black')  # Increased font size

    # Step 5: Add custom labels and title with larger fonts
    ax.set_xticks([2 * i + 0.5 for i in range(n)])
    ax.set_xticklabels([f'{i}' for i in range(n)], fontsize=18)  # Increased font size for tick labels
    if model_name == 'GRAFF':
        model_name_ = 'GRAFF-LP'
    else:
        model_name_ = model_name
    ax.set_title("$||(∇\mathbf{H}^t)_{i,j}||^2$ distribution for " +f"{model_name_} on {dataset_name}", fontsize=24)  # Increased title font size
    ax.set_ylabel("Values (log scale)", fontsize=18)  # Increased ylabel font size

    # Set y-axis to logarithmic scale
    ax.set_yscale('log')

    # Step 6: Add a legend for the colors with larger font
    blue_patch = mpatches.Patch(color='blue', label='Positive edges')
    red_patch = mpatches.Patch(color='red', label='Negative edges')
    ax.legend(handles=[blue_patch, red_patch], loc='lower right', fontsize=18)  # Increased legend font size

    # Step 7: Save the plot as a PDF
    plt.tight_layout()
    plt.savefig(f'gradient_distributions/{model_name}/{dataset_name}/box_plot.pdf', format='pdf', bbox_inches='tight')

    # Step 8: Show the plot (optional)
    plt.show()



#dataset_list = ['Texas', 'Cornell', 'Wisconsin', 'amazon_ratings', 'roman_empire', 'minesweeper', 'questions']
dataset_list = ['tolokers']

models = ['mlp', 'GCN', 'SAGE', 'GAT', 'ELPH']
models_gradient = ['mlp', 'GCN', 'SAGE', 'GAT']

mod1 = 'GRAFF'
metric = 'accuracy'  

# compute_significance('Texas', 'GRAFF', 'GCN', metric = metric)

file_gradient = '_gradient/'
file_hadamard = '/'

file_gradient_nores = '_gradient_nores/'
file_hadamard_nores = '_nores/'

auroc = 'AUROC.npy'
auroc_homo = 'AUROC_homo.npy'
auroc_hetero = 'AUROC_hetero.npy'
auroc_mix_hard = 'AUROC_mix_hard_path.npy'
auroc_mix_easy = 'AUROC_mix_easy_path.npy'

exp_homo = 'exp_homo.npy'
exp_hetero = 'exp_hetero.npy'
exp_mix_hard = 'exp_mix_hard.npy'
exp_mix_easy = 'exp_mix_easy.npy'
exp_tot = 'exp_tot.npy'



auroc_elements = [auroc, auroc_homo, auroc_hetero, auroc_mix_hard, auroc_mix_easy]
exp_elements = [exp_tot]# exp_homo, exp_hetero, exp_mix_hard, exp_mix_easy]
name_list = ['$AUC$', '$AUC_{hm,hm}$', '$AUC_{ht,ht}$', '$AUC_{ht,hm}$', '$AUC_{hm,ht}$']



compare_among_categories(exp_elements, models + [mod1], dataset_list)

display_results_per_metric(file_hadamard, auroc_elements, models + [mod1], dataset = 'tolokers', name_list = name_list)

# display_results_per_datasets(file_hadamard, auroc, models + [mod1], dataset_list)

plot_box_plot(model_name = 'GRAFF', dataset_name = 'tolokers')

compare_among_readouts(auroc_elements, exp_elements, file_hadamard, file_gradient, models + [mod1], dataset_list)
compare_among_readouts(auroc_elements, exp_elements, file_gradient, file_gradient, models + [mod1], dataset_list)


dataframe_gradient = plot_table(dataset_list, models_gradient, mod1, metric = metric, file = file_gradient)
dataframe_hadamard = plot_table(dataset_list, models, mod1, metric = metric, file = file_hadamard)

print("Dataframe Hadamard: \n", dataframe_hadamard)
print("Dataframe Gradient: \n", dataframe_gradient)



