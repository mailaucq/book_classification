from sklearn.manifold import TSNE
import numpy as np

# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range

def visualize_tsne(X, labels):
	tsne = TSNE(n_components=2).fit_transform(X)

	# extract x and y coordinates representing the positions of the images on T-SNE plot
	tx = tsne[:, 0]
	ty = tsne[:, 1]

	tx = scale_to_01_range(tx)
	ty = scale_to_01_range(ty)

	import matplotlib.pyplot as plt
	import itertools as it

	all_classes = sorted(set(labels))
	colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
	colors_per_class = dict(zip(all_classes,it.cycle(colors)))


	# initialize a matplotlib plot
	fig = plt.figure()
	ax = fig.add_subplot(111)

	# for every class, we'll add a scatter plot separately
	for label in colors_per_class:
	    # find the samples of the current class in the data
	    indices = [i for i, l in enumerate(labels) if l == label]

	    # extract the coordinates of the points of this class only
	    current_tx = np.take(tx, indices)
	    current_ty = np.take(ty, indices)

	    # convert the class color to matplotlib format
	    color = colors_per_class[label]

	    # add a scatter plot with the corresponding color and label
	    ax.scatter(current_tx, current_ty, c=color, label=label)

	# build a legend using the labels we set previously
	#ax.legend(loc='best')
	ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


	# finally, show the plot
	plt.show()
