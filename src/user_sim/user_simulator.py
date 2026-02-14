import logging
from typing import List
from .data_extraction import DataExtraction
from .utils.config import errors
from .utils.utilities import *
from .data_gathering import *
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from .utils.config import errors
import logging

from dotenv import load_dotenv
load_dotenv()
# check_keys(["OPENAI_API_KEY"])

client = AzureChatOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_ENDPOINT"],
    api_version=os.environ["OPENAI_API_VERSION"],
    azure_deployment="gpt-4o")

logger = logging.getLogger('Info Logger')

parser = StrOutputParser()

class UserGeneration:

    def __init__(self, user_profile, chatbot, user_id = 1):

        self.user_profile = user_profile
        self.chatbot = chatbot
        self.temp = user_profile.temperature
        self.model = user_profile.model
        # self.user_llm = AzureChatOpenAI(model=self.model, 
        #                                 temperature=self.temp, client=client)

        self.user_llm = client
        
        self.conversation_history = {'interaction': []}
        self.ask_about = user_profile.ask_about.prompt()
        self.ask_about_com = user_profile.ask_about_com.prompt()

        print("ask_about:", self.ask_about)      

        print("ask_about_com:", self.ask_about_com)      

        # NEW - for state tracking

        # Generic 
        self.phrases_per_turn = []
        self.variables_per_turn = []

        # First POI
        self.phrases_all = user_profile.ask_about.phrases
        self.picked_elements_all = user_profile.ask_about.picked_elements
        self.extra_phrased_used = []
        self.used_elements = []

        print("picked_elements:", self.picked_elements_all)
        
        # COM
        self.phrases_all_com = user_profile.ask_about_com.phrases
        self.picked_elements_all_com = user_profile.ask_about_com.picked_elements
        self.extra_phrased_used_com= []
        self.used_elements_com = []

        print("picked_elements_com:", self.picked_elements_all_com)

        print("##############################")
        print("user_id:", user_id)
        print("self.ask_about:", self.ask_about)
        print("##############################")

        self.data_gathering = ChatbotAssistant(user_profile.ask_about.phrases)
        self.user_role_prompt = PromptTemplate(
            input_variables=["reminder", "history"],
            template=self.set_role_template()
        )
        self.goal_style = user_profile.goal_style
        self.test_name = user_profile.test_name
        self.repeat_count = 0
        self.loop_count = 0
        self.interaction_count = 0
        self.user_chain = self.user_role_prompt | self.user_llm | parser
        self.my_context = self.InitialContext()
        self.output_slots = self.__build_slot_dict()
        self.error_report = []
        self.user_id = user_id

        self.applied_com = False

    def get_phrases_for_turn(self,
                             picked_elements_all, 
                             phrases_all,
                             used_elements):
        used_ids = self.pick_elements_for_turn(
            selection_probability=0.5,
            seed=self.user_id,
            picked_elements_all=picked_elements_all,
            used_elements=used_elements,
        )
        print("used_ids", used_ids)
        print("phrases_all", phrases_all)
        phrases_turn = ". ".join([phrases_all[i] for i in used_ids])
        return phrases_turn
    
    def pick_elements_for_turn(
        self,
        picked_elements_all,
        used_elements,
        selection_probability: float = 0.5,
        seed: int | None = None
    ):
        print("********************************")
        print("possible_elements:", picked_elements_all)
        print("used_elements:", used_elements)

        if not picked_elements_all:
            return {}, used_elements.copy(), []

        rng = random.Random(seed)
        selected_dict = {}
        updated_used = used_elements.copy()
        used_ids = []

        # assign internal IDs to possible elements
        element_id_map = {i: elem for i, elem in enumerate(picked_elements_all)}

        # mark already used elements
        used_index_set = set()
        for ue in updated_used:
            for idx, pe in element_id_map.items():
                if ue == pe:
                    used_index_set.add(idx)
                    used_ids.append(idx)
                    break

        # always include the first element if not already used
        if 0 not in used_index_set:
            updated_used.append(picked_elements_all[0])
            used_index_set.add(0)
            used_ids.append(0)

        selected_dict.update(picked_elements_all[0])

        # randomly select remaining elements
        additional_selected = False
        for idx, element in element_id_map.items():
            if idx == 0 or idx in used_index_set:
                continue
            if rng.random() < selection_probability:
                updated_used.append(element)
                used_ids.append(idx)
                selected_dict.update(element)
                additional_selected = True

        # ensure at least one additional element is selected
        if not additional_selected:
            for idx, element in element_id_map.items():
                if idx == 0 or idx in used_index_set:
                    continue
                updated_used.append(element)
                used_ids.append(idx)
                selected_dict.update(element)
                break  # only pick one if needed

        used_elements = updated_used

        def flatten_turn(turn):
            return {k: v for d in turn for k, v in d.items()}

        self.variables_per_turn.append(flatten_turn(used_elements.copy()))

        print("selected_dict:", selected_dict)
        print("updated_variables_per_turn:", self.variables_per_turn)
        print("**************")
        return used_ids

    
    def __build_slot_dict(self):
        slot_dict = {}
        output_list = self.user_profile.output
        for output in output_list:
            var_name = list(output.keys())[0]
            slot_dict[var_name] = None
        return slot_dict

    class InitialContext:
        def __init__(self):
            self.original_context = []
            self.context_list = []

        def initiate_context(self, context):

            default_context = ["never recreate a whole conversation, just act as you're a user or client",
                               "never indicate that you are the user, like 'user: bla bla'",
                               'Sometimes, interact with what the assistant just said.',
                               'Never act as the assistant, always behave as a user.',
                               "Don't end the conversation until you've asked everything you need.",
                               "you're testing a chatbot, so there are random values or irrational things "
                               "in your requests"
                               ]

            if isinstance(context, list):
                self.original_context = context.copy() + default_context.copy()
                self.context_list = context.copy() + default_context.copy()
            else:
                self.original_context = [context] + default_context
                self.context_list = [context] + default_context

            # print("context_list after initiate_context:", self.context_list)
            # input()
        def add_context(self, new_context):
            if isinstance(new_context, list):
                for cont in new_context:
                    self.context_list.append(cont)
            else:
                self.context_list.append(new_context)
                # TODO: add exception to force the user to initiate the context

        def get_context(self):
            return '. '.join(self.context_list)

        def reset_context(self):
            self.context_list = self.original_context.copy()

    def set_role_template(self):
        reminder = """{reminder}"""
        history = """History of the conversation so far: {history}"""
        role_prompt = self.user_profile.role + reminder + history
        return role_prompt

    def repetition_track(self, response, reps=3):
        # TODO integrate change of mind poi for case it fails

        self.my_context.reset_context()
        logger.info(f'Context list: {self.my_context.context_list}')
        print("In repetition track")
        if nlp_processor(response, self.chatbot.fallback, 0.6):
            print("REPETITION CHECK")
            self.repeat_count += 1
            self.loop_count += 1
            logger.info(f"is fallback. Repeat_count: {self.repeat_count}. Loop count: {self.loop_count}")

            # if self.repeat_count >= reps:
            #     self.repeat_count = 0
            #     change_topic = """
            #                    Since the assistant is not understanding what you're saying, change the 
            #                    topic to other things to ask about without starting a new conversation
            #                    """

            #     self.my_context.add_context(change_topic)
            if self.repeat_count > 2:
                # COM required
                return None
            return True
        else:
            self.repeat_count = 0
            self.loop_count = 0
            return False

    @staticmethod
    def conversation_ending(response):
        return nlp_processor(response, "src/testing/user_sim/end_conversation_patterns.yml", 0.5)

    def get_history(self):

        lines = []
        for inp in self.conversation_history['interaction']:
            for k, v in inp.items():
                lines.append(f"{k}: {v}")
        return "\n".join(lines)

    def update_history(self, role, message):
        self.conversation_history['interaction'].append({role: message})

    def get_last_user_input(self):
        utterance = [e['user'] for e in self.conversation_history['interaction'] if 'user' in e][:-1]
        return utterance
    
    def is_last_user_turn_repeated(self) -> bool:
        msgs = [e['user'] for e in self.conversation_history['interaction'] if 'user' in e]
        return len(msgs) > 1 and msgs[-1] == msgs[-2]
    
    def end_conversation(self, input_msg):
        # print("gathering register:\n", self.data_gathering.gathering_register)  # Debugging

        if self.goal_style[0] == 'steps' or self.goal_style[0] == 'random steps':
            if self.interaction_count >= self.goal_style[1]:
                logger.info('is end')
                return True

        elif self.conversation_ending(input_msg) or self.loop_count >= 9:
            errors.append({1000: 'Exceeded loop Limit'})
            logger.warning('Loop count surpassed 9 interactions. Ending conversation.')
            return True

        elif 'all_answered' in self.goal_style[0] or 'default' in self.goal_style[0]:
            # print("gathering register inside end_conversation:\n", self.data_gathering.gathering_register)  # Debugging
            if (self.data_gathering.gathering_register["verification"].all()
                and self.all_data_collected()
                    or self.goal_style[2] <= self.interaction_count):
                logger.info(f'limit amount of interactions achieved: {self.goal_style[2]}. Ending conversation.')
                return True
            else:
                return False

        else:
            return False

    def all_data_collected(self):
        output_list = self.user_profile.output
        for output in output_list:
            var_name = list(output.keys())[0]
            var_dict = output.get(var_name)
            if var_name in self.output_slots and self.output_slots[var_name] is not None:
                continue
            my_data_extract = DataExtraction(self.conversation_history,
                                             var_name,
                                             var_dict["type"],
                                             var_dict["description"])
            value = my_data_extract.get_data_extraction()
            if value[var_name] is None:
                return False
            else:
                self.output_slots[var_name] = value[var_name]
        return True
    
    def update_context_with_new_ask_about(self, used_elements, phrases_all, picked_elements_all):
        interaction_style_prompt = self.get_interaction_styles_prompt()
        
        print("interaction_style_prompt:", interaction_style_prompt)
        ask_about_turn_phrases = self.get_phrases_for_turn(
            used_elements=used_elements,
            phrases_all=phrases_all,
            picked_elements_all=picked_elements_all

        )
        print("ask_about_turn_phrases:", ask_about_turn_phrases)

        self.my_context.initiate_context([self.user_profile.context,
                                          interaction_style_prompt,
                                          ask_about_turn_phrases])
        
        language_context = self.user_profile.get_language()

        self.my_context.add_context(language_context)

        return self.my_context.get_context()

    def get_response(self, input_msg):
    
        self.update_history("Assistant", input_msg)
        self.data_gathering.add_message(self.conversation_history)

        if self.end_conversation(input_msg):
            return "exit"

        if self.repetition_track(input_msg) == True:
            print("simulating repetition of user message")
            print("last variables per tun:", self.variables_per_turn[-1])

                       # else:
            ask_repetition = f"""
                            You are simulating the user. Reformulate the last question you said as a user.
                            Just output the rephrased version, nothing else. Do not output the same utterance.

                            Last question:{{}}
                            """.format(self.get_last_user_input())
            
            self.my_context.initiate_context([self.user_profile.context,
                                                self.get_interaction_styles_prompt(),
                                                ])
                
            language_context = self.user_profile.get_language()

            self.my_context.add_context(language_context)            
            self.my_context.add_context(ask_repetition)

            self.variables_per_turn.append(self.variables_per_turn[-1])  # add empty dict for this turn since no new variables were picke
            
        elif self.repetition_track(input_msg) == None:
            # if repeated more than x times decide for COM

            phrases_all = self.phrases_all_com
            picked_elements_all = self.picked_elements_all_com
            used_elements = self.used_elements_com
            extra_phrased_used = self.extra_phrased_used_com

            self.update_context_with_new_ask_about(
                phrases_all=phrases_all,
                picked_elements_all=picked_elements_all,
                used_elements=used_elements
            )
            self.applied_com = True
        else:

            rand_com = random.random()

            if (rand_com < 0.1 and not self.applied_com):
                phrases_all = self.phrases_all_com
                picked_elements_all = self.picked_elements_all_com
                used_elements = self.used_elements_com
                extra_phrased_used = self.extra_phrased_used_com
                self.applied_com = True
            else:
                phrases_all = self.phrases_all
                picked_elements_all = self.picked_elements_all
                used_elements = self.used_elements
                extra_phrased_used = self.extra_phrased_used

            rand = random.random()

            if rand < 0.7:  
                print("New ask about added")
                # add here new information to ask about
                self.update_context_with_new_ask_about(
                    phrases_all=phrases_all,
                    picked_elements_all=picked_elements_all,
                    used_elements=used_elements
                )
            else:
                # take the phrase which dont require los
                print("Simulating phrase without los")
                num_phrases_extra = len(phrases_all) - len(picked_elements_all)
                if random.randint(0, num_phrases_extra - 1) >= 0:
                    extra_phrase = self.phrases_all[len(picked_elements_all) + len(extra_phrased_used)]
                    self.extra_phrased_used.append(extra_phrase)
                    self.my_context.add_context(extra_phrase)

                    self.variables_per_turn.append({})  # add empty dict for this turn since no new variables were picked

        # self.my_context.add_context(self.user_profile.get_language())

        history = self.get_history()

        # print("##############\ncontext before user response:", self.my_context.get_context())

        user_response = self.user_chain.invoke({'history': history, 'reminder': self.my_context.get_context()})

        self.update_history("User", user_response)

        self.interaction_count += 1

        return user_response

    @staticmethod
    def formatting(role, msg):
        return [{"role": role, "content": msg}]

    def get_interaction_styles_prompt(self):
        interaction_style_prompt = []
        for instance in self.user_profile.interaction_styles:
            if instance.change_language_flag:
                pass
            else:
                interaction_style_prompt.append(instance.get_prompt())
        return ''.join(interaction_style_prompt)

    def open_conversation(self, input_msg=None):

        # self.my_context.initiate_context([self.user_profile.context,
        #                                   interaction_style_prompt,
        #                                   self.ask_about])

        self.update_context_with_new_ask_about(picked_elements_all=self.picked_elements_all,
                                               used_elements=self.used_elements,
                                               phrases_all=self.phrases_all
                                               )

        history = self.get_history()

        if input_msg:
            self.update_history("Assistant", input_msg)
            self.data_gathering.add_message(self.conversation_history)
            if self.end_conversation(input_msg):
                return "exit"
            self.repetition_track(input_msg)
        
        # print("##############\ncontext before user response:", self.my_context.get_context())
        user_response = self.user_chain.invoke({'history': history, 'reminder': self.my_context.get_context()})
        print("user_response", user_response)
        
        self.update_history("User", user_response)

        self.data_gathering.add_message(self.conversation_history)
        self.interaction_count += 1
        return user_response
