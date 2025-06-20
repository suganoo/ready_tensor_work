from typing import Annotated, List, Literal
from pydantic import BaseModel
from operator import add
from pyjokes import get_joke
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph


# ===================
# Define State
# ===================


class Joke(BaseModel):
    text: str
    category: str


class JokeState(BaseModel):
    """
    Represents the evolving state of the joke bot.
    """

    jokes: Annotated[List[Joke], add] = []  # Using built-in add operator
    jokes_choice: Literal["n", "c", "q"] = "n"  # next, change, quit
    category: str = "neutral"
    language: str = "en"
    quit: bool = False


# ===================
# Utilities
# ===================


def get_user_input(prompt: str) -> str:
    return input(prompt).strip().lower()


def print_joke(joke: Joke):
    """Print a joke with nice formatting."""
    # print(f"\n📂 CATEGORY: {joke.category.upper()}\n")
    print(f"\n😂 {joke.text}\n")
    print("=" * 60)


def print_menu_header(category: str, total_jokes: int):
    """Print a compact menu header."""
    print(f"🎭 Menu | Category: {category.upper()} | Jokes: {total_jokes}")
    print("-" * 50)


def print_category_menu():
    """Print a nicely formatted category selection menu."""
    print("📂" + "=" * 58 + "📂")
    print("    CATEGORY SELECTION")
    print("=" * 60)


# ===================
# Define Nodes
# ===================


def show_menu(state: JokeState) -> dict:
    print_menu_header(state.category, len(state.jokes))
    print("Pick an option:")
    user_input = get_user_input(
        "[n] 🎭 Next Joke  [c] 📂 Change Category  [q] 🚪 Quit\nUser Input: "
    )
    while user_input not in ["n", "c", "q"]:
        print("❌ Invalid input. Please try again.")
        user_input = get_user_input(
            "[n] 🎭 Next Joke  [c] 📂 Change Category  [q] 🚪 Quit\n    User Input: "
        )
    return {"jokes_choice": user_input}


def fetch_joke(state: JokeState) -> dict:
    joke_text = get_joke(language=state.language, category=state.category)
    new_joke = Joke(text=joke_text, category=state.category)
    print_joke(new_joke)
    return {"jokes": [new_joke]}  # LangGraph will use the add reducer to append this


def update_category(state: JokeState) -> dict:
    categories = ["neutral", "chuck", "all"]
    print_category_menu()

    for i, cat in enumerate(categories):
        emoji = "🎯" if cat == "neutral" else "🥋" if cat == "chuck" else "🌟"
        print(f"    {i}. {emoji} {cat.upper()}")

    print("=" * 60)

    try:
        selection = int(get_user_input("    Enter category number: "))
        if 0 <= selection < len(categories):
            selected_category = categories[selection]
            print(f"    ✅ Category changed to: {selected_category.upper()}")
            return {"category": selected_category}
        else:
            print("    ❌ Invalid choice. Keeping current category.")
            return {}
    except ValueError:
        print("    ❌ Please enter a valid number. Keeping current category.")
        return {}


def exit_bot(state: JokeState) -> dict:
    print("\n" + "🚪" + "=" * 58 + "🚪")
    print("    GOODBYE!")
    print("=" * 60)
    return {"quit": True}


def route_choice(state: JokeState) -> str:
    """
    Router function to determine the next node based on user choice.
    Keys must match the target node names.
    """
    if state.jokes_choice == "n":
        return "fetch_joke"
    elif state.jokes_choice == "c":
        return "update_category"
    elif state.jokes_choice == "q":
        return "exit_bot"
    else:
        return "exit_bot"


# ===================
# Build Graph
# ===================


def build_joke_graph() -> CompiledStateGraph:
    workflow = StateGraph(JokeState)

    # Register nodes
    workflow.add_node("show_menu", show_menu)
    workflow.add_node("fetch_joke", fetch_joke)
    workflow.add_node("update_category", update_category)
    workflow.add_node("exit_bot", exit_bot)

    # Set entry point
    workflow.set_entry_point("show_menu")

    # Routing logic
    workflow.add_conditional_edges(
        "show_menu",
        route_choice,
        {
            "fetch_joke": "fetch_joke",
            "update_category": "update_category",
            "exit_bot": "exit_bot",
        },
    )

    # Define transitions
    workflow.add_edge("fetch_joke", "show_menu")
    workflow.add_edge("update_category", "show_menu")
    workflow.add_edge("exit_bot", END)

    return workflow.compile()


# ===================
# Main Entry
# ===================


def main():
    print("\n" + "🎉" + "=" * 58 + "🎉")
    print("    WELCOME TO THE LANGGRAPH JOKE BOT!")
    print("    This example demonstrates agentic state flow without LLMs")
    print("=" * 60 + "\n")

    graph = build_joke_graph()

    # print("\n📊 === MERMAID DIAGRAM ===")
    # print(graph.get_graph().draw_mermaid())

    print("\n" + "🚀" + "=" * 58 + "🚀")
    print("    STARTING JOKE BOT SESSION...")
    print("=" * 60)

    final_state = graph.invoke(JokeState(), config={"recursion_limit": 100})

    print("\n" + "🎊" + "=" * 58 + "🎊")
    print("    SESSION COMPLETE!")
    print("=" * 60)
    print(
        f"    📈 You enjoyed {len(final_state.get('jokes', []))} jokes during this session!"
    )
    print(f"    📂 Final category: {final_state.get('category', 'unknown').upper()}")
    print("    🙏 Thanks for using the LangGraph Joke Bot!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()