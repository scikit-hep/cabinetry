import os

import pytfc


if __name__ == "__main__":
    # check whether input data exists
    if not os.path.exists("ntuples/"):
        print("run util/create_histograms.py to create input data")
        raise SystemExit

    # import example config file
    fit_configuration = pytfc.config.read("config_example.yml")
    pytfc.config.print_overview(fit_configuration)

    # create template histograms
    input_folder = "util/ntuples"
    output_folder = "histograms/"
    pytfc.template_builder.create_histograms(fit_configuration, output_folder, only_nominal=True)

    # perform histogram post-processing
    pytfc.template_postprocessor.run(fit_configuration, output_folder)

    # build a workspace
    pass

    # run a fit
    pass

    # visualize some results
    pass
