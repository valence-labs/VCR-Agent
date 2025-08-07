from typing import Any


class StructureExplain():
    """
    Base class for structre explain objects.
    """

    def __init__(self, instances: list[dict[str, Any]], **kwargs):
        self.instances = instances
        self.dag = self.get_dag()
        self.structure_hypothesis = self.get_structure_hypothesis()
        self.paragraph = self.get_paragraph()
        self.question = self.get_question()
        self.perturbation = self.get_perturbation()
        self.raw_response = self.get_raw_response()

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.instances[index]

    def get_dag(self) -> dict[str, Any]:
        """
        Get the DAG of the evaluator.
        """
        dags = [instance['dag'] for instance in self.instances]
        return dags

    def get_structure_hypothesis(self) -> list[str]:
        """
        Get the structure hypothesis of the evaluator.
        """
        structure_hypotheses = [instance['explain'] for instance in self.instances]
        return structure_hypotheses

    def get_paragraph(self) -> list[str]:
        paragraphs = [instance['answer'] for instance in self.instances]
        return paragraphs

    def get_question(self) -> list[str]:
        questions = [instance['question'] for instance in self.instances]
        return questions

    def get_perturbation(self) -> list[str]:
        perturbations = [instance['input_perturbation'] for instance in self.instances]
        return perturbations

    def get_raw_response(self) -> list[str]:
        raw_responses = [instance['raw_response'] for instance in self.instances]
        return raw_responses