cmake_minimum_required(VERSION 3.12) # Проверка версии CMake.
									 # Если версия установленой программы
									 # старее указаной, произайдёт аварийный выход.

add_executable(main main.cpp)		# Создает исполняемый файл с именем main
									# из исходника main.cpp