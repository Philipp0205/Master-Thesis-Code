from learner.reports.correctness import correctness_report
from learner.reports.interpretability import interpretability_report
from learner.reports.relevance import relevance_report
from learner.reports.robustness.robustness import robustness_report
from learner.reports.stability import stability_report
from learner.reports.resources import resource_report


# Create all declared reports
def create_reports(name, names, md, model, y_pred):
    for name in names:
        if name == 'correctness':
            correctness_report(md, model, y_pred)
        elif name == 'relevance':
            relevance_report(md, model, y_pred)
        elif name == 'stability':
            stability_report(md, model, y_pred)
        elif name == 'interpretability':
            interpretability_report(md, model, y_pred)
        elif name == 'resource':
            resource_report(md, model, y_pred)
        elif name == 'robustness':
            robustness_report(name, md, model, y_pred)
        else:
            print(f'Invalid report name: {name}')
