
Review_Template_Prompt = f"""
As a large language model, your task is to analyze batches of 10 product reviews for a single product. 
For each batch, parse the reviews and update the Product Review Aggregated Analysis Template with the combined insights from the previous observations and the new reviews. 
Consider the average ratings, most frequent pros and cons, and the most common improvement suggestions from the reviews while filling the template.

Previous Observations:

{previous_observations}

New Reviews:

{reviews}

---

Product Review Aggregated Analysis Template:
I. Product Information
- Product Name: [Example: Classic Whiteboard]
- Brand: [Example: BoardMaster]
- Category: [Example: Office Supplies]

II. Specifications & Performance
- Main Features: [Example: Magnetic surface, aluminum frame, easy to erase]
- Criteria Ratings (1-5 scale):
  - Ease of Assembly/Installation
  - Functionality
  - Durability
  - Ergonomics
  - Aesthetics
  - Safety
  - Maintenance
  - Sustainability
  - Customization
  - Value for Money

III. Pros and Cons (Top 3)
- Pros:
  - [Example: Easy to install]
  - [Example: Magnetic surface is useful]
  - [Example: Good size for a small office]
- Cons:
  - [Example: Aluminum frame dents easily]
  - [Example: Included marker runs out quickly]
  - [Example: Limited customization options]

IV. Improvement Suggestions (Top 3)
- [Example: Offer a more durable frame option, such as steel or reinforced aluminum.]
- [Example: Include higher-quality, refillable markers to reduce waste and improve the writing experience.]
- [Example: Provide additional size and color options to accommodate varying customer preferences.]

V. Notable Quotes or Experiences

Positive:
[Example: "The magnetic surface is a game-changer for our brainstorming sessions."]
Negative:
[Example: "I was disappointed by how easily the aluminum frame dented during installation."]

"""












TRIZ_Prompts = {
    "0": {
        "instruction": "List the main issues mentioned in the reviews. Analyze the provided product reviews and extract the key issues mentioned by the users. Summarize the main issues in a list format. Save the output with the tag 'issues_list'.",
        "reviews": [
            "The product has a great design, but the battery life is very short.",
            "The user interface is intuitive, but the device is quite heavy.",
            "The device has a lot of useful features, but it takes too long to charge.",
            "The build quality is excellent, but the device is too bulky to carry around.",
            "The price is reasonable, but the device lacks durability."
        ]
    },
    "1": {
        "instruction": "Identify the contradictions between the positive and negative aspects of the product. Based on the main issues identified in the previous response (Recall: 'issues_list'), compare the positive aspects of the product with the negative aspects and identify the contradictions that arise when attempting to improve one aspect while maintaining or enhancing the other. List the identified contradictions. Save the output with the tag 'contradictions_list'."
    },
    "2": {
        "instruction": "Prioritize the contradictions based on their importance to users. Considering the identified contradictions (Recall: 'contradictions_list'), determine which are most critical to users and could significantly impact their satisfaction with the product. Rank the contradictions based on their importance, and provide a brief justification for the ranking. Save the output with the tag 'contradictions_ranking'."
    },
    "3": {
        "instruction": "Determine the relevant parameters for each contradiction using the TRIZ matrix. Based on the contradictions identified and prioritized in Step 1 (Recall: 'contradictions_ranking'), identify the parameters that correspond to the contradictions in the TRIZ matrix (e.g., weight, size, battery life). Look for the intersecting points of the parameters within the matrix. Save the output with the tag 'relevant_parameters'."
    },
    "4": {
        "instruction": "Identify the 40 Inventive Principles that correspond to the selected parameters. Using the relevant parameters determined in the previous response (Recall: 'relevant_parameters'), find the numbers listed at the intersecting points in the TRIZ matrix, which correspond to the 40 Inventive Principles. Review the list of 40 Inventive Principles to understand the corresponding principles. Save the output with the tag 'inventive_principles'."
    },
    "5": {
        "instruction": "Match the identified principles to each contradiction. Apply the appropriate Inventive Principles (Recall: 'inventive_principles') to each contradiction from the prioritized list (Recall: 'contradictions_ranking'). For example, if improving battery life without increasing weight is a contradiction, match the relevant Inventive Principles to that contradiction to generate potential solutions. Save the output with the tag 'principles_to_contradictions'."
    },
    "6": {
        "instruction": "Apply the matched 40 Inventive Principles to the contradictions. Using the Inventive Principles matched to each contradiction in the previous step (Recall: 'principles_to_contradictions'), generate innovative ideas for resolving the contradictions. Consider various combinations of the principles to explore multiple possible solutions. Save the output with the tag 'innovative_ideas'."
    },
    "7": {
        "instruction": "Use the ARIZ (Algorithm for Inventive Problem Solving) to further analyze the problem and generate new ideas. Apply the ARIZ to analyze the problem and generate additional innovative solutions. Begin by formulating a mini-problem statement that captures the essence of the contradiction. Identify available resources and apply the Standard Inventive Solutions and other TRIZ tools to develop and refine potential solutions. Save the output with the tag 'ARIZ_ideas'."
    },
    "8": {
        "instruction": "Perform Su-Field analysis to visualize the problem and find inventive solutions. Create a Substance-Field model by identifying the substances and fields involved in the problem. Apply the 76 Standard Inventive Solutions to the Su-Field model to transform it into an ideal solution. Use the transformed Su-Field model to generate new ideas and insights for resolving the contradiction. Save the output with the tag 'Su_Field_ideas'."
    },
    "9": {
        "instruction": "List all the generated ideas and solutions. Compile a comprehensive list of all the ideas and solutions generated during the previous steps (Recall: 'innovative_ideas', 'ARIZ_ideas', and 'Su_Field_ideas'). Organize the list in a clear and logical manner, grouping similar or related ideas together. Save the output with the tag 'all_ideas'."
    },
    "10": {
        "instruction": "Assess the feasibility of each solution considering technical constraints and resources. Evaluate each idea from the list of all generated ideas (Recall: 'all_ideas') based on its technical feasibility, considering the available resources and any constraints. Eliminate or modify ideas that are not technically feasible or face significant constraints that would hinder their implementation. Save the output with the tag 'feasible_ideas'."
    },

    "11": {
        "instruction": "Evaluate the cost-effectiveness of each solution by estimating its development and production costs. Estimate the development and production costs associated with each remaining idea (Recall: 'feasible_ideas'), taking into account factors such as research and development, prototyping, materials, manufacturing processes, and labor. Assess the cost-effectiveness of each solution and prioritize solutions that offer a good balance between cost and effectiveness. Save the output with the tag 'cost_effective_ideas'."
    },
    "12": {
        "instruction": "Estimate the potential impact of each solution on user experience and satisfaction. Evaluate the potential impact of each solution on user experience, taking into account the user's needs, preferences, and expectations. Consider factors such as usability, ergonomics, aesthetics, performance, reliability, and durability. Prioritize solutions that have a high potential to improve user experience and satisfaction while maintaining or enhancing the product's positive aspects. Save the output with the tag 'high_impact_ideas'."
    },
    "13": {
        "instruction": "Rank the solutions based on their feasibility, cost-effectiveness, and potential impact. Based on the evaluations conducted in Step 4 (Recall: 'feasible_ideas', 'cost_effective_ideas', and 'high_impact_ideas'), rank the solutions according to their overall feasibility, cost-effectiveness, and potential impact on user experience and satisfaction. Create a scoring system or use a decision matrix to help with ranking the solutions objectively. Save the output with the tag 'ranked_solutions'."
    },
    "14": {
        "instruction": "Choose the top-ranked solution(s) for further development. Select the highest-ranked solution(s) from your evaluation (Recall: 'ranked_solutions') for further development. If multiple solutions address different aspects of the product or different contradictions, consider selecting and combining them to create an integrated solution. Save the output with the tag 'selected_solutions'."
    },
    "15": {
        "instruction": "Develop a prototype or mock-up for the selected solution(s). Create a physical or digital prototype or mock-up of the selected solution(s) (Recall: 'selected_solutions'), incorporating the improvements and changes identified during the ideation process. Focus on accurately representing the key features and functionality of the improved product, ensuring that it addresses the contradictions and user needs. Save the output with the tag 'prototypes'."
    },
    "16": {
        "instruction": "Test the prototype(s) with users to gather feedback and validate the solution. Conduct user testing with the prototype(s) (Recall: 'prototypes') to gather feedback on its functionality, usability, and overall user experience. Use a combination of quantitative and qualitative methods to collect comprehensive feedback from users. Analyze the feedback to identify any remaining issues, areas for improvement, or new contradictions that may have arisen as a result of the proposed solution(s). Save the output with the tag 'user_feedback'."
    },
    "17": {
        "instruction": "Refine the prototype(s) based on user feedback and iterate until the desired level of user satisfaction is achieved. Make necessary adjustments to the prototype(s) based on the user feedback (Recall: 'user_feedback') and retest the prototype(s) to ensure that the improvements are effective. Continue iterating on the prototype(s) and conducting user testing until the desired level of user satisfaction is achieved, and the identified contradictions are resolved or minimized. Save the output with the tag 'final_prototype'."
    }
}





Problem_Template_Prompt = f""""As a Large Language Model, 
your task is to break down a main challenge into sub-problems 
and define a clear and specific problem statement for each sub-problem. 
The main challenge is to improve {customer} {experience} {problem} 
for a {product} in the {industry} industry.

Please brainstorm all of the potential sub-problems that are related to the main challenge,
refine and prioritize the list based on their importance and relevance, 
and define a concise problem statement for each sub-problem that outlines the specific issue to be addressed.
""""





TRIZ_40_PRINCIPLES = [
    {"id": 1, "name": "Segmentation", "description": "Divide an object into independent parts."},
    {"id": 2, "name": "Taking out", "description": "Separate an interfering part or property from an object."},
    {"id": 3, "name": "Local quality", "description": "Change an object's structure from uniform to non-uniform."},
    {"id": 4, "name": "Asymmetry", "description": "Change the shape of an object from symmetrical to asymmetrical."},
    {"id": 5, "name": "Merging", "description": "Bring closer together or merge objects that perform similar functions."},
    {"id": 6, "name": "Universality", "description": "Make a part or object perform multiple functions."},
    {"id": 7, "name": "'Nested doll'", "description": "Place one object inside another; place each object in a third one, and so on."},
    {"id": 8, "name": "Anti-weight", "description": "Compensate for the weight of an object by joining it with another."},
    {"id": 9, "name": "Preliminary anti-action", "description": "Perform an action beforehand that will counteract a negative force."},
    {"id": 10, "name": "Preliminary action", "description": "Perform some action before it is needed."},
    {"id": 11, "name": "Beforehand cushioning", "description": "Prepare emergency means beforehand to compensate for the relatively low reliability of an object."},
    {"id": 12, "name": "Equipotentiality", "description": "Change the working conditions so an object need not be raised or lowered."},
    {"id": 13, "name": "Inversion", "description": "Invert the action of an object."},
    {"id": 14, "name": "Spheroidality - Curvature", "description": "Instead of using rectilinear parts, surfaces, or forms, use curved ones."},
    {"id": 15, "name": "Dynamics", "description": "If an object is immobile, make it movable."},
    {"id": 16, "name": "Partial or excessive actions", "description": "Perform a required action partially or in excess."},
    {"id": 17, "name": "Another dimension", "description": "Move an object in two- or three-dimensional space."},
    {"id": 18, "name": "Mechanical vibration", "description": "Cause an object to oscillate or vibrate."},
    {"id": 19, "name": "Periodic action", "description": "Instead of continuous action, use periodic or pulsating actions."},
    {"id": 20, "name": "Continuity of useful action", "description": "Eliminate all idle or intermittent functioning of an object."},
    {"id": 21, "name": "Skipping", "description": "Conduct a process at high speed or with high frequency."},
    {"id": 22, "name": "Blessing in disguise", "description": "Use harmful factors to achieve a positive effect."},
    {"id": 23, "name": "Feedback", "description": "Introduce feedback to improve a process or system by monitoring the output and adjusting the input accordingly."},
    {"id": 24, "name": "Intermediary", "description": "Use an intermediary object to transfer or carry out an action."},
    {"id": 25, "name": "Self-service", "description": "Make an object serve itself by performing auxiliary helpful functions."},
    {"id": 26, "name": "Copying", "description": "Instead of an unavailable, expensive, or fragile object, use simpler and inexpensive copies."},
    {"id": 27, "name": "Cheap short-living objects", "description": "Replace an expensive object with a multiple of inexpensive objects."},
    {"id": 28, "name": "Mechanics substitution", "description": "Replace a mechanical means with a sensory (optical, acoustic, taste, or smell) means."},
    {"id": 29, "name": "Pneumatics and hydraulics", "description": "Use gas and liquid parts of an object instead of solid parts."},
    {"id": 30, "name": "Flexible shells and thin films", "description": "Use flexible shells and thin films instead of three-dimensional structures."},
    {"id": 31, "name": "Porous materials", "description": "Make an object porous or add porous elements."},
    {"id": 32, "name": "Color changes", "description": "Change the color of an object or its external environment."},
    {"id": 33, "name": "Homogeneity", "description": "Make objects interact with a given object of the same material."},
    {"id": 34, "name": "Discarding and recovering", "description": "Make portions of an object that have fulfilled their functions go away or modify them directly during operation."},
    {"id": 35, "name": "Parameter changes", "description": "Change an object's physical state or parameters."},
    {"id": 36, "name": "Phase transitions", "description": "Use phenomena occurring during phase transitions."},
    {"id": 37, "name": "Thermal expansion", "description": "Use thermal expansion or contraction of materials."},
    {"id": 38, "name": "Strong oxidants", "description": "Replace common air with oxygen-enriched air or pure oxygen."},
    {"id": 39, "name": "Inert atmosphere", "description": "Replace a normal environment with an inert one."},
    {"id": 40, "name": "Composite materials", "description": "Change from uniform to composite materials."}]




def get_triz_principles(problem_statement):
    print("Getting TRIZ principles")
    # You can filter or prioritize the principles based on the problem_statement
    keywords = extract_keywords(problem_statement)
    filtered_principles = []

    for principle in TRIZ_40_PRINCIPLES:
        for keyword in keywords:
            if keyword.lower() in principle['description'].lower():
                filtered_principles.append(principle)
                break

    return filtered_principles

def extract_keywords(text):
    print("Extracting keywords")
    # You can replace this with a more sophisticated method for extracting keywords
    stopwords = {'a', 'an', 'the', 'in', 'on', 'at', 'and', 'or', 'of', 'to', 'for', 'with'}
    words = text.split()
    keywords = [word.strip().lower() for word in words if word.strip().lower() not in stopwords]
    return keywords