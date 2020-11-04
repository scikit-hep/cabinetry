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
        print("run util/create_ntuples.py to create input data")
        raise SystemExit

    # import example config file
    cabinetry_config = cabinetry.configuration.load("config_example.yml")
    cabinetry.configuration.print_overview(cabinetry_config)

    # create template histograms
    cabinetry.template_builder.create_histograms(cabinetry_config, method="uproot")

    # perform histogram post-processing
    cabinetry.template_postprocessor.run(cabinetry_config)

    # visualize systematics templates
    cabinetry.visualize.templates(cabinetry_config)

    # build a workspace and save to file
    workspace_path = "workspaces/example_workspace.json"
    ws = cabinetry.workspace.build(cabinetry_config)
    cabinetry.workspace.save(ws, workspace_path)

    # run a fit
    ws = cabinetry.workspace.load(workspace_path)
    fit_results = cabinetry.fit.fit(ws)

    # visualize pulls and correlation matrix
    cabinetry.visualize.pulls(fit_results)
    cabinetry.visualize.correlation_matrix(fit_results)

    # visualize pre- and post-fit distributions
    cabinetry.visualize.data_MC(cabinetry_config, ws)
    cabinetry.visualize.data_MC(cabinetry_config, ws, fit_results=fit_results)
