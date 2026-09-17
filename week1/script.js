"use strict";

const menu = [
  { name: "Pizza", icon: "fa-pizza-slice", tag: "Cheesy & classic" },
  { name: "Burger", icon: "fa-burger", tag: "Hearty & juicy" },
  { name: "Sushi", icon: "fa-fish", tag: "Fresh & light" },
  { name: "Ramen", icon: "fa-bowl-food", tag: "Warm & slurpy" },
  { name: "Soup", icon: "fa-bowl-rice", tag: "Cozy & comforting" },
  { name: "Hot Dog", icon: "fa-hotdog", tag: "Classic & quick" },
  { name: "Fried Chicken", icon: "fa-drumstick-bite", tag: "Crispy & satisfying" },
  { name: "Tacos", icon: "fa-pepper-hot", tag: "Spicy & bold" },
  { name: "Salad", icon: "fa-carrot", tag: "Healthy & crisp" },
  { name: "Toast", icon: "fa-bread-slice", tag: "Quick & easy" },
  { name: "Cheese Platter", icon: "fa-cheese", tag: "Savory & indulgent" },
  { name: "Ice Cream", icon: "fa-ice-cream", tag: "Cool & sweet" },
];

const dishIcon = document.getElementById("dishIcon");
const dishName = document.getElementById("dishName");
const dishTag = document.getElementById("dishTag");
const generateBtn = document.getElementById("generateBtn");
const menuCount = document.getElementById("menuCount");

menuCount.textContent = menu.length;

function pickRandomDish() {
  const index = Math.floor(Math.random() * menu.length);
  return menu[index];
}

function renderDish(dish) {
  dishIcon.className = "fa-solid " + dish.icon + " dish-icon";
  dishName.textContent = dish.name;
  dishTag.textContent = dish.tag;
  dishIcon.classList.remove("spin");
  void dishIcon.offsetWidth;
  dishIcon.classList.add("spin");
}

function generate() {
  renderDish(pickRandomDish());
}

generateBtn.addEventListener("click", generate);

generate();