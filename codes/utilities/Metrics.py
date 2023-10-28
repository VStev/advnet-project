import matplotlib.pyplot as plt

def plot_metrics(metrics):
    # Extract the classes and their corresponding precision, recall, and accuracy
    classes = range(len(metrics['Precision']))
    precision = metrics['Precision']
    recall = metrics['Recall']
    accuracy = metrics['Accuracy']

    # Create subplots
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    # Plot Precision
    axes[0].bar(classes, precision, tick_label=classes)
    axes[0].set_title('Precision')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Precision')

    # Plot Recall
    axes[1].bar(classes, recall, tick_label=classes)
    axes[1].set_title('Recall')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Recall')

    # Plot Accuracy
    axes[2].bar(['Accuracy'], [accuracy])
    axes[2].set_title('Accuracy')
    axes[2].set_ylabel('Accuracy')

    plt.tight_layout()
    plt.show()

def plot_reconstruction_error(rec_error):
    plt.figure(figsize=(8, 6))
    plt.hist(rec_error, bins=50, density=True, alpha=0.6, color='b')
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Density")
    plt.title("Reconstruction Error Distribution")
    plt.show()