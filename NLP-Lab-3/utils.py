import json


def load_json(json_path: str):
    with open(json_path, 'r') as file:
        json_dct = json.load(file)

    return json_dct


def format_report(name: str, id: str, performances: dict) -> str:
    """
    Memformat hasil kalkulasi performa dari keempat model

    Args:
        name (str): nama mahasiswa yang mengerjakan
        id (str): npm mahasiswa yang mengerjakan
        performaces (dict): dictionary dengan key=nama model dan value=performance dari model tsb

    Returns:
        str: report hasil performa yang sudah diformat
    """

    # boleh diubah sesuai kebutuhan, tetapi format outputnya
    # harus tetap sama
    report = f"{name} - {id}\n"
    
    for model_name, performance in performances.items():
        report += f"---------- {model_name} ----------\n"
        report += f"Best Accuracy: {performance['best_acc']}\n"
        report += f"Candidate Accuracy: {performance['cand_acc']}\n"
        report += f"Time: {performance['time']} seconds\n"
    
    return report

