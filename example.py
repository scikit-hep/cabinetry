import os

import cabinetry


if __name__ == "__main__":
    # set up customized log formatting
    cabinetry.set_logging()

    # check whether input data exists
    if not os.path.exists("ntuples/"):
        print("run util/create_ntuples.py to create input data")
        raise SystemExit

    # import example config file
    config = cabinetry.configuration.load("config_example.yml")
    cabinetry.configuration.print_overview(config)

    # create template histograms
    cabinetry.template_builder.create_histograms(config, method="uproot")

    # perform histogram post-processing
    cabinetry.template_postprocessor.run(config)

    # visualize systematics templates
    cabinetry.visualize.templates(config)

    # build a workspace and save to file
    workspace_path = "workspaces/example_workspace.json"
    ws = cabinetry.workspace.build(config)
    cabinetry.workspace.save(ws, workspace_path)

    # run a fit
    ws = cabinetry.workspace.load(workspace_path)
    model, data = cabinetry.model_utils.model_and_data(ws)
    fit_results = cabinetry.fit.fit(model, data)

    # visualize pulls and correlation matrix
    cabinetry.visualize.pulls(fit_results)
    cabinetry.visualize.correlation_matrix(fit_results)

    # obtain pre- and post-fit model predictions
    model_prefit = cabinetry.model_utils.prediction(model)
    model_postfit = cabinetry.model_utils.prediction(model, fit_results=fit_results)

    # show post-fit yield table
    cabinetry.tabulate.yields(model_postfit, data)

    # visualize pre- and post-fit distributions
    cabinetry.visualize.data_mc(model_prefit, data, config=config)
    cabinetry.visualize.data_mc(model_postfit, data, config=config)
