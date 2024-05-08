import PySimpleGUI as sg

def input():
    sg.theme("LightBlue2")
    layout = [  [sg.Text('Settings', font='Any 20')],
                    [sg.Text('support ', justification='center', font='Any 12'), sg.Slider(range=(0,1), resolution=0.1, orientation='h', border_width =2, s=(100,20), key='-sup-')],
                    [sg.Text('ratio      ', font='Any 12') , sg.Slider(range=(0,1), resolution=0.1, orientation='h', border_width =2,s=(100,20), key='-r-')],
                    [sg.Text('Cut-evalution function', justification='center', font='Any 12'),
                     sg.Radio("Cost", "faculty", key='-frequency-', font='Any 12', enable_events=True,default=True),
                     sg.Radio("Reward", "faculty", key='-relation-', font='Any 12', enable_events=True),
                     sg.Radio("Approximate", "faculty", key='-aproximate-', font='Any 12', enable_events=True)
                     ],
                    [sg.Text('Cut-candidate exploration', justification='center', font='Any 12'),
                     sg.Radio("Reachability", "faculty2", key='-reachability-', font='Any 12', enable_events=True,default=True),
                     sg.Radio("Graph Neural Network", "faculty2", key='-gnn-', font='Any 12', enable_events=True)
                     ],
                    [sg.Text('Desirable Log(.xes)   ', font='Any 12'), sg.FileBrowse(key="-Desirable Log-")],
                    [sg.Text('Undesirable Log(.xes)', font='Any 12'), sg.FileBrowse(key="-Undesirable Log-")],
                    [sg.Button('Run', font='Any 20', s=(20,2))]]

    window = sg.Window('Inputs', layout, size=(600, 350), element_justification='c')

    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == "Run":
            break

    window.close()
    
    return float(values["-sup-"]), float(values["-r-"]), values["-Desirable Log-"], values["-Undesirable Log-"], values["-frequency-"], values["-relation-"], values["-aproximate-"], values["-gnn-"]


def output(acc,F1,acc_s,F1_s,fitp,prc,time):
    left_column_text = [
    "align-acc(L^+, L^-, M): ",
    "align-F1-score(L^+, L^-, M): ",
    "trace-acc(L^+, L^-, M): ",
    "trace-F1-score(L^+, L^-, M): ",
    "align-fit(L^+, L^-, M): ",
    "precision(L^+, M): ",
    "Runtime: "
    ]
    right_column_text = [
        acc,
        F1,
        acc_s,
        F1_s,
        fitp,
        prc,
        time
    ]   
    layout =    [
                    [
                        sg.Column([
                            [sg.Text(text, font='Any 14')] for text in left_column_text
                        ], size=(300, 300)),
                        sg.Column([
                            [sg.Text(text, font='Any 14')] for text in right_column_text
                        ], size=(300, 300))
                    ]
                ]

    window = sg.Window("Outputs", layout, size=(600, 300), element_justification='c')
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
    window.close()

