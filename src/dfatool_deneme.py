from ltlf2dfa.Parser import MyParser
from ltlf2dfa.Translator import Translator
from ltlf2dfa.DotHandler import DotHandler

formula = "G(a->b)"
parser = MyParser()
parsed_formula = parser(formula)

print(parsed_formula)

declare_flag = False
translator = Translator(formula)
translator.formula_parser()
translator.translate()
translator.createMonafile(declare_flag)
translator.invoke_mona()

dothandler = DotHandler()
dothandler.modify_dot()
dothandler.output_dot()
