import nbformat

def extract_cells(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    code_cells = []
    markdown_cells = []

    for cell in nb.cells:
        if cell.cell_type == 'code':
            code_cells.append(cell.source)
        elif cell.cell_type == 'markdown':
            markdown_cells.append(cell.source)

    return code_cells, markdown_cells

if __name__ == "__main__":
    notebook_path = '/Users/mm/Desktop/AI_code_generator/ALL_Classification_(Final_V8).ipynb'
    code_cells, markdown_cells = extract_cells(notebook_path)
    print("Code Cells:", code_cells)
    print("Markdown Cells:", markdown_cells)