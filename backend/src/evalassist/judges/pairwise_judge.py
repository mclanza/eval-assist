import logging
from textwrap import dedent
from typing import Literal, cast

from evalassist.judges.batch_parser import BatchRepairParser
from evalassist.judges.utils import (
    build_format_instructions,
    generate_dynamic_pydantic_model,
    get_context_dict,
)
from langchain_core.prompts.prompt import PromptTemplate
from pydantic import BaseModel, Field
from unitxt.inference import InferenceEngine

from .base import BasePairwiseJudge, JudgeDescriptor, UnitxtInferenceEngineMixin
from .types import Criteria, PairwiseInstance, PairwiseInstanceResult

logger = logging.getLogger(__name__)


class PairwiseJudge(BasePairwiseJudge, UnitxtInferenceEngineMixin):
    on_generation_failure: Literal["raise", "random"]
    tie_enabled: bool

    def __init__(
        self,
        inference_engine: InferenceEngine,
        self_consistency: bool | int = False,
        on_generation_failure: Literal["raise", "random"] = "random",
        tie_enabled: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(
            inference_engine=inference_engine,
            self_consistency=self_consistency,
            *args,
            **kwargs,
        )

        if on_generation_failure not in ["raise", "random"]:
            raise ValueError(
                "on_generation_failure must be either 'raise' or 'random'. Received {on_generation_failure} instead."
            )
        self.on_generation_failure = on_generation_failure
        self.tie_enabled = tie_enabled

        if self.self_consistency:
            temp = getattr(self.inference_engine, "temperature", None)
            if temp is not None:
                try:
                    if float(temp) == 0.0:
                        logger.warning(
                            "Self-consistency may not bring any benefit when temperature is 0."
                        )
                except (TypeError, ValueError):
                    logger.debug(
                        "Could not interpret temperature value for self-consistency check."
                    )

    def get_name(self) -> str:
        return f"simple{f'_with_self_consistency_{self.self_consistency}_attempts' if self.self_consistency else ''}"

    def get_descriptor(self) -> JudgeDescriptor:
        judge_descriptor = JudgeDescriptor(self.get_name(), "pairwise", "")
        judge_descriptor.inference_engine_id = self.get_inference_engine_id()
        return judge_descriptor

    def generate_pydantic_model(
        self,
        model_name: str,
        valid_options: list[str],
    ) -> type[BaseModel]:
        def validate_selected_option(cls, value: str) -> str:
            if value not in valid_options:
                raise ValueError(f"value must be one of {valid_options}")
            return value

        field_defs = [
            (
                "explanation",
                str,
                Field(..., description="Step by step explanation of the evaluation"),
                [],
            ),
            (
                "selected_option",
                str,
                Field(
                    ...,
                    description=f"The chosen option. Any of {', '.join(valid_options)}",
                ),
                [validate_selected_option],
            ),
        ]
        model: type[BaseModel] = generate_dynamic_pydantic_model(model_name, field_defs)
        return model

    def _run(
        self,
        instances: list[PairwiseInstance],
        criteria: list[Criteria],
    ) -> list[PairwiseInstanceResult]:
        model_classes = []
        format_instructions_list = []
        prediction_fields = []
        valid_options_list = []
        for instance, criterion in zip(instances, criteria):
            prediction_field = (
                criterion.prediction_field
                if criterion.prediction_field is not None
                else "response"
            ).lower()
            prediction_fields.append(prediction_field)

            valid_options = [
                *[
                    f"{prediction_field}_{i + 1}"
                    for i in range(len(instance.responses))
                ],
                "tie",
            ]
            valid_options_list.append(valid_options)

            klass = self.generate_pydantic_model(
                model_name=f"{criterion.name}_model",
                valid_options=valid_options,
            )
            model_classes.append(klass)
            format_instructions_list.append(build_format_instructions(klass))

        # Build context sections
        context_variables_list = [
            get_context_dict(instance, criterion)
            for instance, criterion in zip(instances, criteria)
        ]
        context_sections = [
            (
                "\n\n## Context\n\n"
                + "\n".join(f"- {k}: {v}" for k, v in c.items())
                + "\n"
            )
            if c
            else ""
            for c in context_variables_list
        ]

        # Build example sections
        def get_example_str(criterion: Criteria) -> str:
            if not criterion.examples:
                return ""
            examples_str = []
            for i, example in enumerate(criterion.examples):
                context = get_context_dict(
                    cast(PairwiseInstance, example.instance), criterion
                )
                context_str = "\n".join(f"- {k}: {v}" for k, v in context.items())
                context_section = (
                    f"\n#### Context\n{context_str}" if context_str else ""
                )
                prediction_section = "#### Responses\n" + "\n".join(
                    f"{criterion.prediction_field if criterion.prediction_field else 'text'}_{idx + 1}: {resp}"
                    for idx, resp in enumerate(
                        cast(PairwiseInstance, example.instance).responses
                    )
                )
                explanation_section = (
                    f"#### Explanation\n{example.explanation}"
                    if example.explanation
                    else ""
                )
                selected_section = f"#### Selected option\n{example.selected_option}"
                examples_str.append(
                    f"### Example {i + 1}:\n{context_section}\n{prediction_section}\n{explanation_section}\n{selected_section}"
                )
            return "\n\n".join(examples_str)

        examples_sections = [get_example_str(c) for c in criteria]

        # Judge description (static)
        judge_description_sections = [
            f"You are an evaluator. You are an expert on comparing {pf} texts based on a criterion."
            for pf in prediction_fields
        ]

        # Tie instructions
        tie_sections = [
            f'\n\n## Tie option\nIf two or more of the {pf} texts are equally good based on the criterion, set the "selected_option" field as "tie".\n'
            if self.tie_enabled
            else ""
            for pf in prediction_fields
        ]

        responses_sections = [
            "\n".join(
                [
                    f"{prediction_field}_{i + 1}: {response}"
                    for i, response in enumerate(instance.responses)
                ]
            )
            for instance, prediction_field in zip(instances, prediction_fields)
        ]

        # Build system and user prompts
        system_template = PromptTemplate(
            input_variables=["judge_description_section", "format_instructions"],
            template=dedent(
                """\
                {judge_description_section}

                You must output only valid JSON with no extra text.
                Use the following schema and formatting rules:
                {format_instructions}
                """
            ),
        )

        user_template = PromptTemplate(
            input_variables=[
                "context_section",
                "examples_section",
                "criteria_name_section",
                "criteria_description",
                "prediction_field",
                "tie_section",
                "responses_section",
                "valid_options",
            ],
            template=dedent(
                """\
                You will be given:
                - **Criterion** (name and description)
                - **Optional context**
                - The **{prediction_field}** texts to evaluate

                ## Required evaluation behavior (follow these precisely):
                1. Read the *criterion* and the *context* carefully.
                2. Compare each candidate {prediction_field} against the criterion and the reference context.
                3. Decide which candidate best satisfies the criterion (or decide tie if two or more candidates are equally good).
                4. Write your reasoning in the "explanation", using clear markdown bullet points that describe why one response is better. Keep it concise and factual.
                5. Set "selected_option" to exactly one of the following values: {valid_options}.

                ## Criteria: {criteria_name_section}
                {criteria_description}
                {context_section}
                {examples_section}

                ## The {prediction_field} texts to evaluate
                {responses_section}
                {tie_section}
                """
            ),
        )

        # Compose final prompts as system + user messages
        prompts = [
            [
                {
                    "role": "system",
                    "content": system_template.format(
                        judge_description_section=jd,
                        format_instructions=fi,
                    ),
                },
                {
                    "role": "user",
                    "content": user_template.format(
                        context_section=ctx,
                        examples_section=ex,
                        criteria_name_section=crit.name if crit.name else "\n",
                        criteria_description=crit.description,
                        prediction_field=pf,
                        tie_section=tie,
                        responses_section=resp,
                        valid_options=", ".join([f'"{v}"' for v in valid_opts]),
                    ),
                },
            ]
            for ctx, ex, crit, pf, tie, resp, jd, fi, valid_opts in zip(
                context_sections,
                examples_sections,
                criteria,
                prediction_fields,
                tie_sections,
                responses_sections,
                judge_description_sections,
                format_instructions_list,
                valid_options_list,
            )
        ]

        dataset = [
            {"source": msgs, "data_classification_policy": ["public"]}
            for msgs in prompts
        ]

        # Call inference engine
        unparsed_responses = cast(list[str], self.inference_engine(dataset=dataset))

        # Parse responses
        parser = BatchRepairParser(
            inference_engine=self.inference_engine,
            max_retries=3,
            on_generation_failure=self.on_generation_failure,
        )

        parsed_responses, parsing_metadatas = parser.parse_all_responses(
            unparsed_responses=unparsed_responses,
            on_failure_default=valid_options_list,
            model_classes=model_classes,
        )

        # Map outputs to internal format
        explanations = [r.explanation for r in parsed_responses]  # type: ignore
        selected_options = [r.selected_option for r in parsed_responses]  # type: ignore
        parsed_selected_options = [
            "tie" if so == "tie" else int(so.split("_")[1]) - 1
            for so in selected_options
        ]

        return [
            PairwiseInstanceResult(
                instance=instance,
                criteria=crit,
                selected_option=sel,
                explanation=exp,
                metadata={
                    **meta,
                    "prompt": prompt,
                    "unparsed_response": unparsed,
                },
            )
            for sel, exp, prompt, unparsed, crit, meta, instance in zip(
                parsed_selected_options,
                explanations,
                prompts,
                unparsed_responses,
                criteria,
                parsing_metadatas,
                instances,
            )
        ]
