%option noyywrap

%{
#define YY_DECL int yylex()

#include "pcfg.tab.h"
%}

%%
[\t]
"*"             {yylval.sval = "PTR\t"; return PTR;}
"array"         {yylval.sval = "array\t"; return ARR;}
"struct"        {yylval.sval = "struct\t"; return STRU;}
"union"         {yylval.sval = "union\t"; return UNION;}
"->"            {yylval.sval = "FUNC\t"; return FUNC;}
"enum"          {yylval.sval = "enum\t"; return ENUM;}
"int"           {yylval.sval = "int\t"; return INT;}
"char"          {yylval.sval = "char\t"; return CHAR;}
"double"        {yylval.sval = "double\t"; return DOUBLE;}
"short"         {yylval.sval = "short\t"; return SHORT;}
"bool"          {yylval.sval = "bool\t"; return BOOL;}
"float"         {yylval.sval = "float\t"; return FLOAT;}
"long"          {yylval.sval = "long\t"; return LONG;}
"longlong"     {yylval.sval = "longlong\t"; return LONGLONG;}
"longdouble"   {yylval.sval = "longdouble\t"; return LONGDOUBLE;}
"void"          {yylval.sval = "void\t"; return VOID;}
"O"             {yylval.sval = "O\t"; return VOID;}
"<eos>"         {return EOS;}
[\n]            {return EOL;}
%%
