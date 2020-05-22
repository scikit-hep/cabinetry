import os
import uproot
import numpy as np
import matplotlib.pyplot as plt


output_directory = "ntuples/"
if not os.path.exists(output_directory):
    os.mkdir(output_directory)


def toy_distribution(noncentral, multiplier, offset, num_events):
    return (
        np.random.noncentral_chisquare(5, noncentral, num_events) * multiplier + offset
    )


def toy_weights(total_yield, num_events):
    avg = total_yield / float(num_events)
    weights = np.random.normal(avg, avg * 0.1, num_events)
    # re-normalize to make sure sum of weights exactly matches avg
    weights *= total_yield / np.sum(weights)
    return weights


def get_samples(num_events):
    dist_s = toy_distribution(10, 12, 350, num_events)
    dist_b = toy_distribution(10, 25, 0, num_events)
    return [dist_s, dist_b]


def get_weights(yield_s, yield_b, num_events):
    w_s = toy_weights(yield_s, num_events)
    w_b = toy_weights(yield_b, num_events)
    return [w_s, w_b]


def create_pseudodata(yield_s, yield_b):
    # create a dataset with some slightly different composition
    scale_s = 1.2
    scale_b = 1.0
    dist_s = toy_distribution(10, 12, 350, int(yield_s * scale_s))
    dist_b = toy_distribution(10, 25, 0, int(yield_b * scale_b))
    pseudodata = np.hstack((dist_s, dist_b))
    np.random.shuffle(pseudodata)
    return pseudodata


def create_lepton_charge(n_events):
    # lepton charge is +1 or -1 for all events, just to have an extra variable
    charge = (np.random.randint(0, 2, size=n_events) * 2) - 1
    return charge


def create_file(file_name, distributions, weights, labels):
    n_events = len(weights[0])
    with uproot.recreate(file_name) as f:
        # write the predicted processes
        for i, label in enumerate(labels):
            lep_charge = create_lepton_charge(n_events)
            f[label] = uproot.newtree(
                {"jet_pt": "float64", "weight": "float64", "lep_charge": "int"}
            )
            f[label].extend(
                {
                    "jet_pt": distributions[i],
                    "weight": weights[i],
                    "lep_charge": lep_charge,
                }
            )


def create_file_pseudodata(file_name, pseudodata):
    n_events = len(pseudodata)
    with uproot.recreate(file_name) as f:
        # write pseudodata
        lep_charge = create_lepton_charge(n_events)
        f["pseudodata"] = uproot.newtree({"jet_pt": "float64", "lep_charge": "int"})
        f["pseudodata"].extend({"jet_pt": pseudodata, "lep_charge": lep_charge})


def read_file(file_name):
    distributions = []
    weights = []
    labels = []
    with uproot.open(file_name) as f:
        all_trees = f.allkeys(
            filterclass=lambda cls: issubclass(cls, uproot.tree.TTreeMethods)
        )
        for tree in all_trees:
            distributions.append(f[tree].array("jet_pt"))
            weights.append(f[tree].array("weight"))
            labels.append(tree)
    return distributions, weights, labels


def read_file_pseudodata(file_name):
    with uproot.open(file_name) as f:
        distribution = f["pseudodata"].array("jet_pt")
    return distribution


def plot_distributions(data, weights, labels, pseudodata, bins):
    bin_width_str = str(int(bins[1] - bins[0]))

    # labels = [l.split('\'')[1] for l in labels]
    yield_each = [str(round(np.sum(w), 1)) for w in weights]
    labels = [label.decode().split(";")[0] for label in labels]

    # plot normalized distributions
    for i in reversed(range(len(data))):
        plt.hist(
            data[i],
            weights=weights[i],
            bins=bins,
            label=labels[i],
            histtype="step",
            density=True,
        )
    plt.legend(frameon=False)
    plt.xlabel(r"jet $p_T$ [GeV]")
    plt.ylabel("normalized")
    plt.savefig("normalized.png", dpi=200)

    # plot stack
    plt.clf()
    labels_with_yield = [labels[i] + " " + yield_each[i] for i in range(len(labels))]
    pseudodata_label = "pseudodata " + str(len(pseudodata))
    plt.hist(
        data[::-1],
        weights=weights[::-1],
        bins=bins,
        label=labels_with_yield[::-1],
        histtype="stepfilled",
        stacked=True,
    )
    plt.hist(pseudodata, bins=bins, label=pseudodata_label, histtype="step", color="k")
    plt.legend(frameon=False)
    plt.xlabel(r"jet $p_T$ [GeV]")
    plt.ylabel("events / " + bin_width_str + " GeV")
    plt.savefig("stacked.png", dpi=200)


if __name__ == "__main__":
    # configuration
    num_events = 1000
    yield_s = 125
    yield_b = 1000
    labels = ["signal", "background"]  # names of prcesses
    file_name = output_directory + "prediction.root"
    file_name_pseudodata = output_directory + "data.root"

    np.random.seed(0)

    # distributions for two processes
    distributions = get_samples(num_events)

    # corresponding weights
    weights = get_weights(yield_s, yield_b, num_events)

    # create a pseudodataset
    pseudodata = create_pseudodata(yield_s, yield_b)

    # write it all to a file
    create_file(file_name, distributions, weights, labels)
    create_file_pseudodata(file_name_pseudodata, pseudodata)

    # read the files again
    d_read, w_read, l_read = read_file(file_name)
    pd_read = read_file_pseudodata(file_name_pseudodata)

    # compare predictions from before/after reading
    np.testing.assert_allclose(d_read, distributions)
    np.testing.assert_allclose(w_read, weights)
    np.testing.assert_allclose(pd_read, pseudodata)

    # visualize results
    bins = np.linspace(0, 1200, 24 + 1)
    plot_distributions(d_read, w_read, l_read, pseudodata, bins)
