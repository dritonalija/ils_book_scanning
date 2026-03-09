import os
import sys


def read_input_file(input_path):
    with open(input_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    b_count, l_count, day_limit = map(int, lines[0].strip().split())
    book_scores = list(map(int, lines[1].strip().split()))
    libraries = []
    for i in range(2, len(lines), 2):
        if i + 1 >= len(lines):
            break
        n_books, signup_days, books_per_day = map(int, lines[i].strip().split())
        books = set(map(int, lines[i + 1].strip().split()))
        libraries.append((n_books, signup_days, books_per_day, books))
    return b_count, l_count, day_limit, book_scores, libraries


def read_output_file(output_path):
    with open(output_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    num_libraries = int(lines[0].strip())
    solution = []
    index = 1
    for _ in range(num_libraries):
        if index >= len(lines):
            break
        lib_id, num_books = map(int, lines[index].strip().split())
        index += 1
        books = list(map(int, lines[index].strip().split())) if index < len(lines) else []
        index += 1
        solution.append((lib_id, num_books, books))
    return num_libraries, solution


def validate_solution(input_path, output_path, is_console_application=False):
    b_count, l_count, day_limit, book_scores, libraries = read_input_file(input_path)
    num_libraries, solution = read_output_file(output_path)

    errors = []
    all_scanned_books = set()
    used_libraries = set()
    total_days_used = 0
    total_score = 0

    with open(output_path, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file.readlines()]

    expected_library_count = len(lines[1:]) // 2
    if num_libraries != expected_library_count:
        errors.append(
            f"Invalid solution: Declared {num_libraries} libraries, but output contains {expected_library_count} library entries."
        )
    if num_libraries > l_count:
        errors.append(f"Invalid solution: Output references {num_libraries} libraries, but only {l_count} exist.")

    for lib_id, num_books, books in solution:
        if lib_id >= l_count:
            errors.append(f"Library {lib_id} does not exist.")
            continue
        if lib_id in used_libraries:
            errors.append(f"Library {lib_id} is listed multiple times in the solution.")
            continue

        n_books, signup_days, books_per_day, library_books = libraries[lib_id]
        if total_days_used + signup_days >= day_limit:
            errors.append(f"Library {lib_id} takes too long to sign up ({signup_days} days), leaving no time for scanning.")
            continue

        total_days_used += signup_days
        used_libraries.add(lib_id)

        if num_books != len(books):
            errors.append(
                f"Library {lib_id}: Declared {num_books} books, but actually listed {len(books)} books in output file."
            )

        invalid_books = [b for b in books if b not in library_books]
        if invalid_books:
            errors.append(f"Library {lib_id} contains invalid books: {invalid_books}.")

        unique_books = [b for b in books if b not in all_scanned_books]
        if len(unique_books) != len(books):
            errors.append(f"Library {lib_id} contains duplicate global book assignments.")
        all_scanned_books.update(unique_books)

        remaining_days = day_limit - total_days_used
        max_possible_books = min(remaining_days * books_per_day, len(library_books))
        if len(books) > max_possible_books:
            errors.append(
                f"Library {lib_id} attempts to scan {len(books)} books, exceeding the limit of {max_possible_books}."
            )

        total_score += sum(book_scores[b] for b in unique_books if 0 <= b < len(book_scores))

    if is_console_application:
        return "Valid" if not errors else "Invalid"

    if errors:
        return "\n".join(errors)

    return f"Solution is valid!\nTotal score: {total_score}\n"


def main():
    if len(sys.argv) == 3:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        result = validate_solution(input_path, output_path, True)
        print(result)
        return

    try:
        from PyQt6.QtWidgets import (
            QApplication,
            QWidget,
            QVBoxLayout,
            QPushButton,
            QLabel,
            QFileDialog,
            QTextEdit,
            QSpacerItem,
            QSizePolicy,
        )
    except ModuleNotFoundError:
        print("PyQt6 is required for GUI mode. Use: python validator\\validator.py <input> <output>")
        return

    class ValidatorApp(QWidget):
        def __init__(self):
            super().__init__()
            self.init_ui()

        def init_ui(self):
            self.setWindowTitle("Hash Code 2020 Validator")
            self.setGeometry(100, 100, 600, 400)
            layout = QVBoxLayout()

            self.input_label = QLabel("Input file: None")
            layout.addWidget(self.input_label)
            self.input_button = QPushButton("Browse Input File")
            self.input_button.clicked.connect(self.browse_input)
            layout.addWidget(self.input_button)

            self.output_label = QLabel("Output file: None")
            layout.addWidget(self.output_label)
            self.output_button = QPushButton("Browse Output File")
            self.output_button.clicked.connect(self.browse_output)
            layout.addWidget(self.output_button)

            spacer = QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
            layout.addItem(spacer)

            self.validate_button = QPushButton("Validate Solution")
            self.validate_button.clicked.connect(self.validate)
            layout.addWidget(self.validate_button)

            self.result_text = QTextEdit()
            self.result_text.setReadOnly(True)
            layout.addWidget(self.result_text)

            self.setLayout(layout)
            self.input_path = None
            self.output_path = None

        def browse_input(self):
            file_name, _ = QFileDialog.getOpenFileName(self, "Open Input File", "", "Text Files (*.txt)")
            if file_name:
                self.input_path = file_name
                self.input_label.setText(f"Input file: {os.path.basename(file_name)}")

        def browse_output(self):
            file_name, _ = QFileDialog.getOpenFileName(self, "Open Output File", "", "Text Files (*.txt)")
            if file_name:
                self.output_path = file_name
                self.output_label.setText(f"Output file: {os.path.basename(file_name)}")

        def validate(self):
            if not self.input_path or not self.output_path:
                self.result_text.setText("Please select both input and output files.")
                return
            result = validate_solution(self.input_path, self.output_path)
            self.result_text.setText(result)

    app = QApplication(sys.argv)
    validator = ValidatorApp()
    validator.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
