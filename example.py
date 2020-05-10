import logging
import os

import cabinetry


# set up log formatting and suppress verbose output from matplotlib
logging.basicConfig(
    level=logging.DEBUG, format="%(levelname)s - %(name)s - %(message)s"
)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


if __name__ == "__main__":
    # check whether input data exists
    if not os.path.exists("ntuples/"):
        print("run util/create_histograms.py to create input data")
        raise SystemExit

    # import example config file
    fit_configuration = cabinetry.config.read("config_example.yml")
    cabinetry.config.print_overview(fit_configuration)

    # create template histograms
    input_folder = "util/ntuples"
    histogram_folder = "histograms/"
    cabinetry.template_builder.create_histograms(
        fit_configuration, histogram_folder, only_nominal=True, method="uproot"
    )

    # perform histogram post-processing
    cabinetry.template_postprocessor.run(fit_configuration, histogram_folder)

    # build a workspace and save to file
    workspace_folder = "workspaces/"
    workspace_name = "example_workspace"
    ws = cabinetry.workspace.build(fit_configuration, histogram_folder)
    cabinetry.workspace.save(ws, workspace_folder, workspace_name)

    # run a fit
    pass

    # visualize some results
    figure_folder = "figures/"
    cabinetry.visualize.data_MC(
        fit_configuration,
        histogram_folder,
        figure_folder,
        prefit=True,
        method="matplotlib",
    )
