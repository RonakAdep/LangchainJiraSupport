from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
)
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.output_parsers import PydanticOutputParser
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from typing import Literal
from langchain_classic.schema.runnable import (
    RunnableBranch,
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
    RunnableSequence,
)
import json

# model setup
# for task
# 1. Original ticket
# 2. Extracted information
# 3. Ticket category
# 4. Priority

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite", temperature=0.3, max_tokens=2048
)


# Schema setup
class analysis_data(BaseModel):
    Employee_issue: str = Field(
        description="A brief description of the main technical problem or issue reported by the employee."
    )
    Affected_system: str = Field(
        description="The specific hardware, software, application, network, or system affected by the reported issue."
    )
    Symptons: str = Field(
        description="A detailed description of the observable problems, errors, or unusual behaviour experienced by the employee due to the reported issue."
    )
    bussiness_impact: Literal["Critical", "High", "Medium", "Low"] = Field(
        description="The level of impact the reported issue has on the employees work or the companys business operations. Classify it as Critical, High, Medium, or Low."
    )
    deadline_urgency: Literal["Immediate", "4 hours", "24 hours", "Flexible"] = Field(
        description="The urgency by which the issue needs to be addressed based on the employees stated deadline and the time sensitivity of the issue. Choose Immediate, 4 hours, 24 hours, or Flexible."
    )


class ticket_classification(BaseModel):
    issue_type: Literal[
        "Hardware", "Software", "Network", "Database", "Security", "Other"
    ] = Field(
        description="""Identify the primary category of the reported issue, such as 
            physical hardware failures, software or application errors, 
            network connectivity problems, database-related issues, 
            security incidents, or issues that do not fit these categories."""
    )

    ticket_priority: Literal["Critical", "High", "Medium", "Low"] = Field(
        description="""Determine the urgency of the ticket based on issue severity, 
            business impact, and service disruption. Consider the number 
            of affected users, operational impact, and availability of 
            temporary workarounds."""
    )


class technical_analysis(BaseModel):
    Technical_Issue: str = Field(
        description="A clear explanation of the technical problem being experienced, including the observed behaviour or error."
    )
    Possible_Root_Cause: str = Field(
        description="The most likely technical reasons behind the issue, based on the reported symptoms and available information."
    )
    Affected_Components: str = Field(
        description="The hardware, software, applications, services, or system components that are directly or potentially affected."
    )
    Troubleshooting_Steps: str = Field(
        description="A practical sequence of steps to investigate, diagnose, and potentially resolve the issue."
    )
    Technical_Severity: str = Field(
        description="The technical severity of the issue based on its impact on system functionality, performance, reliability, and availability."
    )


class bussiness_analysis(BaseModel):
    Employee_Productivity: str = Field(
        description="Explain how the reported issue affects the employee's ability to perform daily tasks and maintain normal productivity."
    )
    Business_Operations: str = Field(
        description="Describe the impact of the issue on business activities, critical processes, services, and overall operational continuity."
    )
    Work_Disruption: str = Field(
        description="Determine whether the employee can continue working normally, work with limitations, or is completely unable to perform their responsibilities."
    )
    Business_Impact: Literal["Critical", "High", "Medium", "Low"] = Field(
        description="Assess the overall business impact based on productivity loss, operational disruption, affected users, and the importance of the interrupted activities. Choose Critical, High, Medium, or Low."
    )
    Recommended_ResponseTime: Literal[
        "Immediate", "4 hours", "24 hours", "Flexible"
    ] = Field(
        description="Recommend how quickly IT support should respond based on business impact and operational urgency. Choose Immediate, 4 hours, 24 hours, or Flexible."
    )


# 1.initial template and praser:
main_template = ChatPromptTemplate(
    [
        (
            "user",
            """You are an IT support ticket analysis assistant.
               Analyse the employee ticket and extract the required information.
                 \n Employee ticket:
                        {user_issue}
                  \n{format_instruction}    
        """,
        ),
    ]
)
parser = PydanticOutputParser(pydantic_object=analysis_data)
main_prompt = main_template.partial(format_instruction=parser.get_format_instructions())

# technical analysis prompt preparations
tech_parser = PydanticOutputParser(pydantic_object=technical_analysis)
tech_prompt = PromptTemplate(
    template="""
You are an IT Technical Analyst. Analyze the following employee issue:
\n
{input_data}
\n
Identify the technical issue, possible root causes, affected components, recommended troubleshooting steps, and technical severity.
Base your analysis only on the provided information. Avoid assumptions and keep the explanations concise.
\n
{format_instruction}
""",
    input_variables=["input_data"],
    partial_variables={"format_instruction": tech_parser.get_format_instructions()},
)


# Bussiness analysis prompt preparations
buss_parser = PydanticOutputParser(pydantic_object=bussiness_analysis)
buss_prompt = PromptTemplate(
    template="""
You are an IT Business Impact Analyst. Analyze the extracted employee issue information provided below:

{input_data}
\n
Evaluate how the issue affects employee productivity, business operations, and the employee's ability to perform their responsibilities.
Based on your analysis, independently determine the overall business impact and recommend an appropriate IT response time.
Use only the provided information. Avoid assumptions and provide clear, concise, professional explanations.
\n
{format_instruction}
""",
    input_variables=["input_data"],
    partial_variables={"format_instruction": buss_parser.get_format_instructions()},
)


#
ticket_parser = PydanticOutputParser(pydantic_object=ticket_classification)
ticket_prompt = PromptTemplate(
    template="""
You are an IT Ticket Classification Specialist.

Analyze the following technical and business analysis information:

Technical Analysis:
{technical_analysis}

Business Impact Analysis:
{bussiness_impact}

Classify the issue_type based on the technical issue and affected components.
Determine the ticket_priority based on technical severity, business impact, and operational urgency.

Provide accurate classifications based on the given information. Avoid assumptions.

{format_instruction}
""",
    input_variables=["technical_analysis", "bussiness_impact"],
    partial_variables={"format_instruction": ticket_parser.get_format_instructions()},
)


# parser
str_parser = StrOutputParser()

hardware_template = PromptTemplate(
    template="""
You are an IT Hardware Support Specialist.
Analyze the following ticket information:
{all_three_comnbine}
\n
{ticket_classification}
\n
Identify the likely hardware problem, recommend diagnostic steps, and provide practical repair or replacement suggestions. Consider device performance, physical components, and hardware reliability.
Provide concise, technically accurate recommendations.
""",
    input_variables=["all_three_comnbine", "ticket_classification"],
)
software_template = PromptTemplate(
    template="""
You are an IT Software Support Specialist.
Analyze the following ticket information:
{all_three_comnbine}
\n
{ticket_classification}
\n
Identify possible application or software issues, recommend troubleshooting steps, and suggest appropriate fixes involving software configuration, updates, compatibility, or reinstallation where necessary.
Provide concise, technically accurate recommendations.
""",
    input_variables=["all_three_comnbine", "ticket_classification"],
)

network_template = PromptTemplate(
    template="""
You are an IT Network Support Specialist.
Analyze the following ticket information:
{all_three_comnbine}
\n
{ticket_classification}
\n
Identify possible network connectivity or configuration issues. Recommend diagnostic steps involving network access, DNS, IP configuration, connectivity, and network services where relevant.
Provide practical and concise troubleshooting recommendations.
""",
    input_variables=["all_three_comnbine", "ticket_classification"],
)

database_template = PromptTemplate(
    template="""
You are an IT Database Support Specialist.
Analyze the following ticket information:

{all_three_comnbine}
\n
{ticket_classification}
\n
Identify possible database-related issues and recommend diagnostic steps involving database availability, connectivity, performance, queries, and service health where relevant.
Suggest appropriate resolution actions while considering data integrity and availability.
Provide concise, technically accurate recommendations.
""",
    input_variables=["all_three_comnbine", "ticket_classification"],
)

security_template = PromptTemplate(
    template="""
You are an IT Security Support Specialist.
Analyze the following ticket information:

{all_three_comnbine}
\n
{ticket_classification}
\n
Identify potential security risks and recommend appropriate investigation, containment, and recovery actions. Consider unauthorized access, suspicious activity, malware indicators, and data exposure where relevant.
Prioritize protecting organizational systems and sensitive information. Do not recommend actions that could destroy evidence.
Provide concise and security-focused recommendations.
""",
    input_variables=["all_three_comnbine", "ticket_classification"],
)

finalformat_prompt = PromptTemplate(
    template="""
You are an IT Support Report Formatter.
Format the following ticket information into a clear, well-structured report:

Ticket Analysis:
{all_info}

Recommended Resolution:
{resolution}

Present all information with appropriate headings, bullet points, and spacing.
Preserve the original information and meaning. Do not add assumptions, explanations, or unnecessary content.
Return only the formatted report.
    """,
    input_variables=["all_info", "resolution"],
)

# transforming function
# def transform_func(x):
#     data = x
#     if hasattr(data, "model_dump"):
#         data = data.model_dump()
#     elif hasattr(data, "dict"):
#         data = data.dict()


#     return json.dumps(data, indent=2)
def transform_func(data):
    def convert(obj):
        if hasattr(obj, "model_dump"):
            return obj.model_dump(mode="json")

        if isinstance(obj, dict):
            return {key: convert(value) for key, value in obj.items()}

        return obj

    return json.dumps(convert(data), indent=2)


def prepare_resolution_input(x):
    return {
        "all_three_comnbine": transform_func(x["all_three_comnbine"]),
        "ticket_classification": transform_func(x["ticket_classification"]),
    }


# chain 1  take input from user and extract information for the text given by user
chain1 = RunnableSequence(main_prompt, llm, parser)

# chain 2  user the extracted information to analyze technical and bussiness impact analysis
chain2 = RunnableParallel(
    {
        "analysis_data": RunnablePassthrough(),
        "technical_analysis": RunnableSequence(
            {"input_data": RunnableLambda(transform_func)},
            tech_prompt,
            llm,
            tech_parser,
        ),
        "bussiness_impact": RunnableSequence(
            {"input_data": RunnableLambda(transform_func)},
            buss_prompt,
            llm,
            buss_parser,
        ),
    }
)

# chai 3 now  it will classify the ticket from techincal analysis and  bussiness analysis
chain3 = RunnableParallel(
    {
        "all_three_comnbine": RunnablePassthrough(),
        "ticket_classification": RunnableSequence(
            {
                "technical_analysis": RunnableLambda(
                    lambda x: transform_func(x["technical_analysis"])
                ),
                "bussiness_impact": RunnableLambda(
                    lambda x: transform_func(x["bussiness_impact"])
                ),
            },
            ticket_prompt,
            llm,
            ticket_parser,
        ),
    }
)

# chain 4 doing conditional processing  after identification based on issue type

software_chain = software_template | llm | str_parser
hardware_chain = hardware_template | llm | str_parser
network_chain = network_template | llm | str_parser
database_chain = database_template | llm | str_parser
security_chain = security_template | llm | str_parser


chain4 = RunnableParallel(
    {
        "all_info": RunnablePassthrough(),
        "resolution": RunnableBranch(
            (
                lambda x: x["ticket_classification"].issue_type == "Hardware",
                RunnableLambda(prepare_resolution_input) | hardware_chain,
            ),
            (
                lambda x: x["ticket_classification"].issue_type == "Software",
                RunnableLambda(prepare_resolution_input) | software_chain,
            ),
            (
                lambda x: x["ticket_classification"].issue_type == "Network",
                RunnableLambda(prepare_resolution_input) | network_chain,
            ),
            (
                lambda x: x["ticket_classification"].issue_type == "Database",
                RunnableLambda(prepare_resolution_input) | database_chain,
            ),
            (
                lambda x: x["ticket_classification"].issue_type == "Security",
                RunnableLambda(prepare_resolution_input) | security_chain,
            ),
            RunnableLambda(lambda x: (print("this is the default chain"), x)[1]),
        ),
    }
)

chain5 = RunnableSequence(
    {
        "all_info"  : RunnableLambda(lambda x: transform_func(x["all_info"])),
        "resolution": RunnableLambda(lambda x: transform_func(x["resolution"])),
    },
    finalformat_prompt,
    llm,
    str_parser,
)
# connecting chain1 and chain2

final_res = RunnableSequence(chain1, chain2, chain3, chain4, chain5)
result = final_res.invoke(
    {
        "user_issue": "My laptop has become extremely slow since yesterday. Applications take a long time to open, and sometimes the system freezes completely. I have an important client presentation tomorrow"
    }
)

print("\n")
print(result)
