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
    fit_config = cabinetry.config.read("config_example.yml")
    cabinetry.config.print_overview(fit_config)

    # create template histograms
    input_folder = "util/ntuples"
    histo_folder = "histograms/"
    cabinetry.template_builder.create_histograms(
        fit_config, histo_folder, only_nominal=True, method="uproot"
    )

    # perform histogram post-processing
    cabinetry.template_postprocessor.run(fit_config, histo_folder)

    # build a workspace and save to file
    workspace_folder = "workspaces/"
    workspace_name = "example_workspace"
    ws = cabinetry.workspace.build(fit_config, histo_folder)
    cabinetry.workspace.save(ws, workspace_folder, workspace_name)

    # run a fit
    cabinetry.fit.fit(ws)

    # visualize some results
    figure_folder = "figures/"
    cabinetry.visualize.data_MC(
        fit_config, histo_folder, figure_folder, prefit=True, method="matplotlib",
    )
