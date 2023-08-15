%{
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int yylex();
extern int yyparse();
extern int yylex_destroy();
extern FILE* yyin;

void yyerror(const char* s);
char *catstr(const char* a, const char* b);

// FILE *out;
%}

%union{
    char *sval;
}

%token EOL EOS
%token<sval> PTR ARR STRU UNION FUNC ENUM
%token<sval> INT CHAR DOUBLE SHORT BOOL FLOAT LONG LONGLONG LONGDOUBLE VOID O

%type<sval> D S E T N

%start D

%%
D: N EOL            {$$ = catstr($1, "\n"); printf("%s", $$); $$ = "";YYACCEPT;}
  | N error EOL     {$$ = catstr($1, "\n"); printf("%s", $$); $$ = "";YYACCEPT;}
  /* | EOL             {printf("please enter: \n");} */

N: T N              {$$ = catstr($1, $2);}
  | S               {$$ = $1;}


S:  PTR E                 {$$ = catstr($1, $2);}
  | ARR E                 {$$ = catstr($1, $2);}
  | STRU E                {$$ = catstr($1, $2);}
  | UNION E               {$$ = catstr($1, $2);}
  | FUNC E                {$$ = catstr($1, $2);}
  | ENUM E                {$$ = catstr($1, $2);}
  | T                     {$$ = $1;}

E: S E              {$$ = catstr($$, $2);}
  | EOS             {$$ = "<eos>\t";}

T: INT              {$$ = $1;}
  | CHAR            {$$ = $1;}
  | DOUBLE          {$$ = $1;}
  | SHORT           {$$ = $1;}
  | BOOL            {$$ = $1;}
  | FLOAT           {$$ = $1;}
  | LONG            {$$ = $1;}
  | LONGLONG        {$$ = $1;}
  | LONGDOUBLE      {$$ = $1;}
  | VOID            {$$ = $1;}
  | O               {$$ = $1;}

%%

int main(int argc, char **argv){
    /* FILE *f = fopen(argv[1], "r");
    if(!f){
        printf("Bad input: Can not find the file.\n");
        return -1;
    }
    out = fopen(argv[2], "w");
    yyin = f; */
    yyin = stdin;
    yyparse();

    return 0;
}

void yyerror(const char *s){
    //fprintf(stderr, "error: %s\n", s);
}

char* catstr(const char* a, const char* b){
    char *temp = (char *) malloc(strlen(a) + strlen(b) + 1);;
    strcpy(temp, a);
    strcat(temp, b);
    return temp;
}
