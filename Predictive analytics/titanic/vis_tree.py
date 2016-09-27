import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.externals.six import StringIO
from sklearn import tree
import pydotplus as pydot

def render_data(data, saveto):
    graph = pydot.graph_from_dot_data(data)

    # if saveto is None then plot the graph
    if saveto is None:
        # render pydot by calling dot, no file saved to disk
        png_str = graph.create_png(prog='dot')

        # treat the dot output string as an image file
        sio = StringIO()
        sio.write(png_str)
        sio.seek(0)
        img = mpimg.imread(sio)

        # plot the image
        plt.imshow(img, aspect='equal')
        plt.show()
    else:
        graph.write_pdf(saveto)


def render_sklearn_tree(model, feature_names, target_names, saveto=None):
    # feature names: informative names of the features in order.
    # target names: names assigned to integer class labels; Names are matched
    #               in the order to sorted integer class labelsl
    # saveto: location where to save plot; If none, then the plot is shown

    dot_data = StringIO()
    tree.export_graphviz(model, out_file=dot_data,
                              feature_names=feature_names,
                              class_names=target_names,
                              filled=True, rounded=True,
                              special_characters=True)

    render_data(dot_data.getvalue(), saveto)

def render_tree(nodes, edges, saveto=None):
    # This method is not necessary for the IntroDS exercise.
    # It was intended to give as an exercise to come up with tree
    # training code where this method would have been used for rendering,
    # but this is left for the Advanced Data Science course

    # header for the graphviz script
    data = """
    digraph Tree {
    node [shape=box] ;
    """

    # firstly define all the nodes in the tree
    for node_idx, node_label in nodes:
        data += str(node_idx) + ' [label="' + node_label + '"];\n'

    # then define all the connections between nodes in the tree
    for parent, child, label in edges:
        data += str(parent) + ' -> ' + str(child) + ' [labeldistance=2.5, headlabel="' + label + '"];\n'

    data += "}"

    render_data(data, saveto)


if __name__ == "__main__":
    # example tree data.
    # Every node in the tree has unique index, and text of the node.
    nodes = [(0, 'x > 0'), (1, 'y < 1.0'), (2, 'z = 1'), (3, 'w is even')]

    # Edges are defined as set of index of parent node, index of child, label
    edges = [(0, 1, 'T'), (0, 2, 'F'), (2, 3, 'T')]

    # this is how to plot the tree
    render_tree(nodes, edges)

    # this is how to save high quality plot in pdf
    render_tree(nodes, edges, "test.pdf")