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
        print(f"run util/create_histograms.py to create input data")
        raise SystemExit

    # import example config file
    cabinetry_config = cabinetry.configuration.read("config_example.yml")
    cabinetry.configuration.print_overview(cabinetry_config)

    # create template histograms
    histo_folder = "histograms/"
    cabinetry.template_builder.create_histograms(
        cabinetry_config, histo_folder, method="uproot"
    )

    # perform histogram post-processing
    cabinetry.template_postprocessor.run(cabinetry_config, histo_folder)

    # build a workspace and save to file
    workspace_path = "workspaces/example_workspace.json"
    ws = cabinetry.workspace.build(cabinetry_config, histo_folder)
    cabinetry.workspace.save(ws, workspace_path)

    # run a fit
    ws = cabinetry.workspace.load(workspace_path)
    cabinetry.fit.fit(ws)

    # visualize templates and data
    figure_folder = "figures/"
    cabinetry.visualize.data_MC(
        cabinetry_config, histo_folder, figure_folder, prefit=True, method="matplotlib"
    )
