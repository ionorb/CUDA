# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    Makefile                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: yoel <yoel@student.42.fr>                  +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/01/19 18:59:58 by gamoreno          #+#    #+#              #
#    Updated: 2023/09/03 17:04:10 by yoel             ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

### Compilation ###

CC      = cc
NVCC    = nvcc

NVCCFLAGS = -g -G#-Xcudafe --display_error_number
# FLAGS = -pthread -g3 #-fdump-rtl-expand 
# FLAGS  = -Wall -Werror -Wextra 
#FLAGS = -pthread -g3 -Ofast -flto#-pg -A -Iincludes #-Ofast -flto #-march=native -mtune=native -fno-plt -fno-stack-protector -fomit-frame-pointer -fno-asynchronous-unwind-tables -fno-unwind-tables -fno-asynchronous-unwind-tables -fno-unwind-tables -fno-ident -fno-st
# FLAGS  = -Wall -Werror -Wextra -Ofast -flto -pthread
# FLAGS  = -Ofast -flto -pthread
### Executable ###
#-Ofast -flto 
NAME   = cudaRT

### Includes ###

OBJ_PATH  = objs/
HEADER = includes/
SRC_PATH  = sources/
# MLX = libs/minilibx-linux
# LIBFT = libs/libft
CUDA = -I/usr/local/cuda/include -L/usr/local/cuda/lib
INCLUDES = -I $(HEADER) $(CUDA)

### Source Files ###
CORE_DIR	=	core/
CORE		=	main.cu \

OBJ_DIRS	+=	$(addprefix	$(OBJ_PATH),$(CORE_DIR))

SOURCES		+=	$(addprefix	$(CORE_DIR),$(CORE))

### Objects ###

SRCS = $(addprefix $(SRC_PATH),$(SOURCES))
OBJS = $(addprefix $(OBJ_PATH),$(SOURCES:.cu=.o))
DEPS = $(addprefix $(OBJ_PATH),$(SOURCES:.cu=.d))

### COLORS ###
NOC         = \033[0m
GREEN       = \033[1;32m
CYAN        = \033[1;36m

### RULES ###

all: header tmp $(NAME)

tmp:
	@mkdir -p $(OBJ_DIRS)

$(NAME): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(FLAGS) $(INCLUDES) -o $@ $^
	@echo "$(GREEN)Project compiled succesfully$(NOC)"

$(OBJ_PATH)%.o: $(SRC_PATH)%.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@
	@echo "$(CYAN)Creation of object file -> $(CYAN)$(notdir $@)... $(GREEN)[Done]$(NOC)"

clean:
	@echo "$(GREEN)Cleaning libraries files$(NOC)"
	@rm -rf $(OBJ_PATH)

fclean:
	@echo "$(GREEN)Cleaning all$(NOC)"
	@rm -rf $(OBJ_PATH)
	@rm -f $(NAME)

header:
	clear
	@echo "$$HEADER_PROJECT"

re: fclean all

run: all
	clear
	valgrind --leak-check=full --show-leak-kinds=all -s ./$(NAME) test.rt

memcheck: all
	clear
	compute-sanitizer --tool memcheck --leak-check=full ./cudaRT

-include $(DEPS)

.PHONY: tmp, re, fclean, clean, run, memcheck

define HEADER_PROJECT
done
endef
export HEADER_PROJECT
# @echo "$(CYAN)                                                                                                            $(NOC)"
# @echo "$(CYAN)██████╗ ██████╗ ███████╗███████╗███████╗███████╗███████╗███████╗███████╗██████╗ ███████╗███████╗███████╗███████╗███████╗$(NOC)"
# @echo "$(CYAN)██╔══██╗██╔══██╗██╔════╝██╔════╝██╔════╝██╔════╝██╔════╝██╔════╝██╔══██╗██╔════╝██╔════╝██╔════╝██╔════╝██╔════╝██╔════╝$(NOC)"
# @echo "$(CYAN)██████╔╝██████╔╝█████╗  ███████╗███████╗█████╗  ███████╗███████╗███████╗██████╔╝█████╗  ███████╗███████╗███████╗█████╗  $(NOC)"
# @echo "$(CYAN)██╔═══╝ ██╔══██╗██╔══╝  ╚════██║╚════██║██╔══╝  ╚════██║╚════██║╚════██║██╔═══╝ ██╔══╝  ╚════██║╚════██║╚════██║██╔══╝  $(NOC)"
# @echo "$(CYAN)██║     ██║  ██║███████╗███████║███████║███████╗███████║███████║███████║██║     ███████╗███████║███████║███████╗███████╗$(NOC)"
# @echo "$(CYAN)╚═╝     ╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝╚══════╝╚══════╝╚══════╝╚══════╝╚═╝     ╚══════╝╚══════╝╚══════╝╚══════╝╚══════╝$(NOC)"
#                   ░░░░▒▒░░                                            
#             ░░▒▒▒▒░░▒▒▒▒▒▒▒▒▒▒                                        
#           ▓▓▒▒▒▒▒▒▒▒░░░░▒▒▒▒▒▒▒▒░░                                    
#         ▓▓▒▒▒▒▒▒▒▒▒▒░░░░▒▒▒▒▒▒▒▒▓▓░░                                  
#       ▓▓▓▓▒▒▒▒▒▒▒▒▒▒░░░░░░▒▒▒▒▒▒▒▒▓▓                                  
#     ░░██▓▓▓▓▒▒▒▒▒▒▒▒░░░░░░░░░░▒▒▒▒▓▓▓▓                                
#     ▓▓██▓▓▓▓▒▒▒▒▒▒░░░░░░░░░░░░▒▒▒▒▒▒▓▓                                
#     ████▓▓▓▓▓▓▒▒▒▒░░      ░░░░▒▒▒▒▒▒▓▓░░                              
#     ████▓▓▓▓▓▓▓▓▒▒░░      ░░░░▒▒▒▒▓▓▓▓▒▒                              
#     ██████▓▓▓▓▓▓▒▒░░      ░░░░▒▒▒▒▓▓▓▓▒▒                              
#     ██████████▓▓▒▒░░░░  ░░░░▒▒▒▒▒▒▓▓▓▓▒▒                              
#     ██████████▓▓▒▒▒▒░░░░░░▒▒▒▒▓▓▓▓▓▓▓▓░░                              
#     ▓▓████████▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▓▓▓▓▓▓▓▓                                
#     ░░████████▓▓▓▓▒▒▒▒▒▒▒▒▒▒▓▓▓▓▓▓▓▓▒▒                                
#       ▓▓██████▓▓▓▓▓▓▓▓▒▒▓▓▓▓▓▓▓▓▓▓▓▓                                  
#         ██████▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░                                  
#         ░░████████▓▓▓▓▓▓▓▓▓▓▓▓██░░                                    
#             ▓▓████████████████░░                                      
#                 ▒▒▓▓██▓▓▒▒                                            

# ███╗   ███╗██╗███╗   ██╗██╗██████╗ ████████╗
# ████╗ ████║██║████╗  ██║██║██╔══██╗╚══██╔══╝
# ██╔████╔██║██║██╔██╗ ██║██║██████╔╝   ██║   
# ██║╚██╔╝██║██║██║╚██╗██║██║██╔══██╗   ██║   
# ██║ ╚═╝ ██║██║██║ ╚████║██║██║  ██║   ██║   
# ╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝   ╚═╝   
