from learner.reports import *
from learner.reports.correctness import correctness_report
from learner.reports.relevance import relevance_report
from learner.reports.stability import stability_report


# Create all declared reports
def create_reports(names, md, model, y_pred):
    for name in names:
        if name == 'correctness':
            correctness_report(md, model, y_pred)
        elif name == 'relevance':
            relevance_report(md, model, y_pred)
        elif name == 'stability':
            stability_report(md, model, y_pred)
        else:
            print(f'Invalid report name: {name}')
