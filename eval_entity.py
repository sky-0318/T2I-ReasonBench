from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import csv
import json
import re

def extract_json(text):
    # Use a regular expression to find the JSON part
    json_pattern = r'\{.*?\}'
    match = re.search(json_pattern, text, re.DOTALL)

    if match:
        json_string = match.group(0)  # Extract the matched JSON string
        try:
            # Parse the JSON string
            json_data = json.loads(json_string)
            # print("Extracted JSON:", json_data)
        except json.JSONDecodeError as e:
            print("Failed to decode JSON:", e)
    else:
        print("No JSON found in the text.")
    return json_data

def ask_qw(messages, processor, model):
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text

if __name__ == "__main__":

    # default: Load the model on the available device(s)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        # "Qwen/Qwen2.5-VL-7B-Instruct", 
        "Qwen/Qwen2.5-VL-72B-Instruct", 
        torch_dtype="auto", 
        device_map="auto"
    )

    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2.5-VL-7B-Instruct",
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="auto",
    # )

    # default processor
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct")

    # The default range for the number of visual tokens per image in the model is 4-16384.
    # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
    # min_pixels = 256*28*28
    # max_pixels = 1280*28*28
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
    
    prompt_folder = "prompt_folder"
    prompt_txt = f"{prompt_folder}/Entity-Reasoning_prompts_rewrite.txt"
    
    model_names = [
        "my_model"
    ]
        
    for model_name in model_names:   
        
        image_folder = "image_folder"
        csv_path = f"csv_path/{model_name}_entity.csv"
        
        
        with open(prompt_txt, 'r') as file:
            prompts = [line.strip() for line in file]
            
        if os.path.exists(csv_path):
            with open(csv_path, 'r', newline='') as csvreader: 
                reader = csv.reader(csvreader)
                lines = list(reader)  # Read all lines into a list
                line_count = len(lines)  # Count the number of lines
        else:
            line_count = 0
            
        with open(csv_path, 'a', newline='') as csvfile:
            # Create a CSV writer
            csv_writer = csv.writer(csvfile)
            if line_count == 0:
                # Write the header row
                csv_writer.writerow(["id","prompt","answer_1", "answer_2", "score_alignment", "answer_3", "score_quality", "score"])    
                
            all_images = [f for f in os.listdir(image_folder) if f[0].isdigit()]
            all_images = sorted(all_images)
            print(len(all_images))
            
            evaluated = max(line_count - 1,0)
            
            for i in range(evaluated,len(all_images)):
                image_name = all_images[i]
                num = int(image_name[0:4])-1
                prompt = prompts[num]
                
                print(image_name, prompt)
            

                image_path = os.path.join(image_folder, image_name)
                    
                q1 = "Describe this image."   
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image_path,
                            },
                            {"type": "text", "text": q1},
                        ],
                    }
                ]
                
                out1 = ask_qw(messages, processor, model)[0]
                # print(out1)

        
                print(prompt)
                q2 = f"""Given the prompt "{prompt}", please rate the alignment between the image and the prompt on a scale of 1 to 5 according to the criteria:
    5 - Perfectly alignment: the image faithfully captures all key elements of the prompt (subject, setting, time period, distinguishing features) with no inaccuracies.
    4 - Mostly accurate: the image depicts core elements correctly but has minor errors (e.g., slight anachronisms, missing details, or incomplete context).
    3 - Partially correct: the image includes some relevant elements but mixes in inaccuracies (e.g., wrong context, missing critical details, or moderate deviations from the prompt).
    2 - Weak representation: the image only loosely connected to the prompt, with significant inaccuracies (e.g., wrong subject identity, era or location).
    1 - Completely inaccurate: the image fails to reflect the prompt’s core theme, details, or context (e.g., unrelated subject, fantasy elements, or contradictory visuals).
    Please carefully examine the image and check if all the details in the prompt are correctly addressed in the image.
    Provide your answer in json format, including the score and a brief explanation of your reasoning: {{"score": [1 to 5], "explanation": "[within 50 words]"}}"""
        
                new_messages_alignment = messages + [
                    {
                        "role": "assistant",
                        "content": out1,
                        # "content": [
                        #     {
                        #         "type": "text",
                        #         "text": out1[0],
                        #     },
                        # ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", 
                            "text": q2},
                        ],
                    }
                ]
                # print(new_messages)
                
                out2 = ask_qw(new_messages_alignment, processor, model)[0]
                # print(out2)
                json_data_2 = extract_json(out2)
                score_alignment = json_data_2['score']
                # print(score_alignment)
                
                
                
                
                
                q3 = f"""Please rate the image quality on a scale of 1 to 5 according to the criteria:
    5 - Outstanding: Exceptional quality with flawless aesthetics (composition, color, lighting), no technical errors (blur, artifacts), and ethically appropriate (no toxic, biased, or harmful content).
    4 - Very Good: Visually appealing with minor aesthetic/technical flaws (e.g., slight noise, imperfect framing) but no ethical concerns.
    3 - Acceptable: Average aesthetics (generic composition, muted colors) and noticeable flaws, but content is inoffensive.
    2 - Subpar: Distracting flaws (artifacts, deformation) or awkward composition; content risks mild discomfort (e.g., unsettling imagery) but not explicitly harmful.
    1 - Unacceptable: Technically broken (blurry, corrupted) or contains toxic/illegal content (violence, bias) or causes severe psychological harm.
    Provide your answer in json format, including the score and a brief explanation: {{"score": [1 to 5], "explanation": "[within 50 words]"}}"""
        
                new_messages_quality = messages + [
                    {
                        "role": "assistant",
                        "content": out1,
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", 
                            "text": q3},
                        ],
                    }
                ]
                # print(new_messages)
                
                out3 = ask_qw(new_messages_quality, processor, model)[0]
                # print(out2)
                json_data_3 = extract_json(out3)
                score_quality = json_data_3['score']
                # print(score_quality)
                
                score = 0.5*score_alignment + 0.5*score_quality
                print(score_alignment, score_quality, score)
                
                csv_writer.writerow([image_name,prompt,out1,out2,score_alignment,out3,score_quality,score])
                csvfile.flush()